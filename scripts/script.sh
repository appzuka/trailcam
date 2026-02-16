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

def auth_header():
    # PATs are JWT refresh tokens; legacy tokens are static.
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

def replace_label(obj, old="Bird", new="Pigeon"):
    if isinstance(obj, list):
        return [replace_label(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: replace_label(v, old, new) for k, v in obj.items()}
    return new if obj == old else obj

page = 1
page_size = 100
total = 1
updated = 0

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
            new_res = []
            changed = False
            for r in res:
                v = r.get("value")
                if v is None:
                    new_res.append(r)
                    continue
                new_v = replace_label(v)
                if new_v != v:
                    r = dict(r)
                    r["value"] = new_v
                    changed = True
                new_res.append(r)

            if changed:
                api_patch(f"/api/annotations/{ann['id']}/", {"result": new_res})
                updated += 1

    page += 1

print(f"Updated annotations: {updated}")
PY
