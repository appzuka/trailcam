#!/usr/bin/env bash
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

RECT_FROM_NAME = "label"        # RectangleLabels name
CHOICE_FROM_NAME = "bird_species"  # Choices name
TAX_FROM_NAME = "bird_tax"       # Taxonomy name
TO_NAME = "image"                # Image name

TARGET_LABEL = "Bird"
FORCED_CHOICE = "Pigeon"


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

            # Remove all taxonomy results for this annotation
            new_res = []
            for r in res:
                if r.get("type") == "taxonomy" and r.get("from_name") == TAX_FROM_NAME:
                    changed = True
                    continue
                new_res.append(r)

            # Index existing choices by region id
            choice_idx = {
                r.get("id"): i
                for i, r in enumerate(new_res)
                if r.get("type") == "choices"
                and r.get("from_name") == CHOICE_FROM_NAME
                and r.get("id")
            }

            # Force choice = Pigeon for Bird regions
            for r in new_res:
                if r.get("type") != "rectanglelabels":
                    continue
                if r.get("from_name") != RECT_FROM_NAME:
                    continue
                value = r.get("value")
                labels_info = labels_list(value)
                if not labels_info:
                    continue
                labels, _ = labels_info
                if TARGET_LABEL not in labels:
                    continue

                region_id = r.get("id")
                if not region_id:
                    continue

                choice_value = {"choices": [FORCED_CHOICE]}
                if region_id in choice_idx:
                    cr = dict(new_res[choice_idx[region_id]])
                    cr["value"] = choice_value
                    new_res[choice_idx[region_id]] = cr
                else:
                    new_res.append({
                        "id": region_id,
                        "from_name": CHOICE_FROM_NAME,
                        "to_name": TO_NAME,
                        "type": "choices",
                        "value": choice_value,
                    })
                changed = True
                updated_regions += 1

            if changed:
                api_patch(f"/api/annotations/{ann['id']}/", {"result": new_res})
                updated_annots += 1

    page += 1

print(f"Updated annotations: {updated_annots}")
print(f"Updated regions: {updated_regions}")
PY
