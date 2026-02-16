#!/usr/bin/env bash

# This script removes tasks that have been skipped and moves the images for those tasks into the unassigned folder

# This script must be run in the container hosting Label Studio

set -euo pipefail

export LS_URL="${LS_URL:-http://localhost:8080}"
: "${LS_TOKEN:?Set LS_TOKEN in your environment}"
export LS_TOKEN
export LS_PROJECT="${LS_PROJECT:-2}"

export IMAGES_ROOT="/data/local-files/images"
export DEST_DIR="/data/local-files/images/unassigned"
export PAGE_SIZE="200"
export DRY_RUN="0"  # set to 1 to preview

python - <<'PY'
import os, json, urllib.request, urllib.parse, shutil

LS_URL = os.environ["LS_URL"].rstrip("/")
TOKEN = os.environ["LS_TOKEN"]
PROJECT = os.environ["LS_PROJECT"]
IMAGES_ROOT = os.environ.get("IMAGES_ROOT", "/data/local-files/images")
DEST_DIR = os.environ.get("DEST_DIR", "/data/local-files/images/unassigned")
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", "200"))
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"


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


def resolve_path(value):
    if not isinstance(value, str) or not value:
        return None

    if value.startswith("file://"):
        return urllib.parse.urlparse(value).path

    if value.startswith("/data/"):
        parsed = urllib.parse.urlparse(value)
        if parsed.path.startswith("/data/local-files/"):
            q = urllib.parse.parse_qs(parsed.query)
            if "d" in q and q["d"]:
                rel = q["d"][0].lstrip("/")
                if rel.startswith("local-files/"):
                    rel = rel[len("local-files/") :]
                return os.path.join("/data/local-files", rel)
        return value

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urllib.parse.urlparse(value)
        if parsed.path.startswith("/data/local-files/"):
            if parsed.query:
                q = urllib.parse.parse_qs(parsed.query)
                if "d" in q and q["d"]:
                    rel = q["d"][0].lstrip("/")
                    if rel.startswith("local-files/"):
                        rel = rel[len("local-files/") :]
                    return os.path.join("/data/local-files", rel)
            return parsed.path
        q = urllib.parse.parse_qs(parsed.query)
        if "d" in q and q["d"]:
            rel = q["d"][0].lstrip("/")
            if rel.startswith("local-files/"):
                rel = rel[len("local-files/") :]
            return os.path.join("/data/local-files", rel)
        return None

    if value.startswith("images/"):
        return os.path.join("/data/local-files", value)

    return None


def iter_data_values(data):
    if isinstance(data, dict):
        for v in data.values():
            yield v
    elif isinstance(data, list):
        for v in data:
            yield v


def collect_used_paths():
    used = set()
    page = 1
    total = 1
    while (page - 1) * PAGE_SIZE < total:
        data = api_get("/api/tasks/", {
            "project": PROJECT,
            "fields": "all",
            "page": page,
            "page_size": PAGE_SIZE,
        })
        tasks = data.get("tasks") or data.get("results") or []
        total = data.get("total") or data.get("count") or len(tasks)

        for task in tasks:
            tdata = task.get("data") or {}
            for v in iter_data_values(tdata):
                path = resolve_path(v)
                if path:
                    used.add(os.path.normpath(path))
        page += 1
    return used


def move_file(path):
    if not os.path.exists(path):
        return False, f"missing: {path}"

    if os.path.commonpath([IMAGES_ROOT, path]) == IMAGES_ROOT:
        rel = os.path.relpath(path, IMAGES_ROOT)
        dest = os.path.join(DEST_DIR, rel)
    else:
        dest = os.path.join(DEST_DIR, os.path.basename(path))

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(path, dest)
    return True, f"moved: {path} -> {dest}"


used = collect_used_paths()

moved = 0
scanned = 0

for root, _, files in os.walk(IMAGES_ROOT):
    # Skip destination directory if it's inside IMAGES_ROOT
    if os.path.commonpath([DEST_DIR, root]) == DEST_DIR:
        continue
    for name in files:
        path = os.path.normpath(os.path.join(root, name))
        scanned += 1
        if path in used:
            continue
        if DRY_RUN:
            print(f"DRY_RUN: would move {path}")
            continue
        ok, msg = move_file(path)
        print(msg)
        if ok:
            moved += 1

print(f"Scanned files: {scanned}")
print(f"Moved files: {moved}")
PY
