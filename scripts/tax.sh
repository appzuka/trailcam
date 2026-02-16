#!/usr/bin/env bash

# This script is an example of how to change the taxonomy of existing annotations

set -euo pipefail

export LS_URL="${LS_URL:-http://localhost:8080}"
: "${LS_TOKEN:?Set LS_TOKEN in your environment}"
export LS_TOKEN
export LS_PROJECT="${LS_PROJECT:-2}"

python - <<'PY'
import os, json, urllib.request, urllib.parse

LS_URL = os.environ["LS_URL"].rstrip("/")
TOKEN = os.environ["LS_TOKEN"]
PROJECT = os.environ["LS_PROJECT"]

OLD_LABEL = "Pigeon"
NEW_LABEL = "Bird"
TAXONOMY_PATH = ["Bird", "Pigeon"]
RECT_FROM_NAME = "label"   # RectangleLabels name in your config
TAX_FROM_NAME = "bird_tax" # Taxonomy name in your config
TO_NAME = "image"          # Image name in your config

def auth_header():
    if TOKEN.count(".") == 2:
        try:
            data = json.dumps({"refresh": TOKEN}).encode("utf-8")
            req = urllib.request.Request(
                LS_URL + "/api/token/refresh",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as r:
                access = json.load(r).get("access")
            if access:
                return {"Authorization": f"Bearer {access}"}
        except Exception:
            return {"Authorization": f"Bearer {TOKEN}"}
    return {"Authorization": f"Token {TOKEN}"}

AUTH = auth_header()


def api_get(path, params=None):
    url = LS_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=AUTH)
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def api_patch(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LS_URL + path,
        data=data,
        headers={**AUTH, "Content-Type": "application/json"},
        method="PATCH",
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def labels_list(value):
    if not isinstance(value, dict):
        return None
    if "rectanglelabels" in value and isinstance(value["rectanglelabels"], list):
        return value["rectanglelabels"], "rectanglelabels"
    if "labels" in value and isinstance(value["labels"], list):
        return value["labels"], "labels"
    return None

page = 1
page_size = 100
total = 1
updated_annots = 0
updated_regions = 0

while (page - 1) * page_size < total:
    data = api_get("/api/tasks/", {
        "project": PROJECT,
        "fields": "all",
        "only_annotated": "true",
        "page": page,
        "page_size": page_size,
    })
    tasks = data.get("tasks") or data.get("results") or []
    total = data.get("total") or data.get("count") or len(tasks)

    for task in tasks:
        annots = task.get("annotations")
        if annots is None:
            annots = api_get(f"/api/tasks/{task['id']}/annotations/")
        if isinstance(annots, str):
            annots = json.loads(annots)

        for ann in annots:
            res = ann.get("result") or []
            changed = False

            # Map result id to taxonomy result index (if any)
            tax_by_id = {}
            for idx, r in enumerate(res):
                if r.get("type") == "taxonomy" and r.get("from_name") == TAX_FROM_NAME:
                    tax_by_id[r.get("id")] = idx

            for idx, r in enumerate(res):
                if r.get("type") != "rectanglelabels":
                    continue
                if r.get("from_name") != RECT_FROM_NAME:
                    continue
                value = r.get("value")
                labels_info = labels_list(value)
                if not labels_info:
                    continue
                labels, key = labels_info
                if OLD_LABEL not in labels:
                    continue

                # Replace Pigeon -> Bird in rectangle labels
                new_labels = [NEW_LABEL if l == OLD_LABEL else l for l in labels]
                if new_labels != labels:
                    r = dict(r)
                    new_value = dict(value)
                    new_value[key] = new_labels
                    r["value"] = new_value
                    res[idx] = r
                    changed = True
                    updated_regions += 1

                # Ensure taxonomy result exists for the same region id
                region_id = r.get("id")
                if region_id:
                    tax_value = {"taxonomy": [TAXONOMY_PATH]}
                    if region_id in tax_by_id:
                        tr_idx = tax_by_id[region_id]
                        tr = dict(res[tr_idx])
                        tr["value"] = tax_value
                        res[tr_idx] = tr
                    else:
                        res.append({
                            "id": region_id,
                            "from_name": TAX_FROM_NAME,
                            "to_name": TO_NAME,
                            "type": "taxonomy",
                            "value": tax_value,
                        })
                    changed = True

            if changed:
                api_patch(f"/api/annotations/{ann['id']}/", {"result": res})
                updated_annots += 1

    page += 1

print(f"Updated annotations: {updated_annots}")
print(f"Updated regions: {updated_regions}")
PY
