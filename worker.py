import os
import time
import json
import requests
import numpy as np
import cv2
from datetime import datetime, timezone

print("=== HDR WORKER v3 (BLACK FIX STABLE) LOADED ===")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
RAW_BUCKET = os.getenv("RAW_BUCKET", "hdr_raw")
OUT_BUCKET = os.getenv("OUT_BUCKET", "hdr_output")

HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
}

MAX_LONG_EDGE = 2200
JPEG_QUALITY = 92

# ------------------------
# Supabase helpers
# ------------------------

def safe_json(r):
    if not r.text:
        return {}
    try:
        return r.json()
    except:
        return {}

def sb_get(path):
    r = requests.get(f"{SUPABASE_URL}{path}", headers=HEADERS)
    r.raise_for_status()
    return safe_json(r)

def sb_patch(table, row_id, payload):
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/{table}?id=eq.{row_id}",
        headers={**HEADERS, "Prefer": "return=representation"},
        data=json.dumps(payload),
    )
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return safe_json(r)

# ------------------------
# Storage
# ------------------------

def download_image(path):
    r = requests.get(
        f"{SUPABASE_URL}/storage/v1/object/{RAW_BUCKET}/{path}",
        headers={"Authorization": HEADERS["Authorization"]},
    )
    r.raise_for_status()
    arr = np.frombuffer(r.content, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def upload_image(path, img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Encode failed")

    r = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/{OUT_BUCKET}/{path}",
        headers={
            "Authorization": HEADERS["Authorization"],
            "Content-Type": "image/jpeg",
            "x-upsert": "true",
        },
        data=buf.tobytes(),
    )
    if r.status_code >= 400:
        raise RuntimeError(r.text)

# ------------------------
# HDR
# ------------------------

def resize_safe(img):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= MAX_LONG_EDGE:
        return img
    scale = MAX_LONG_EDGE / m
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def exposure_fusion(imgs):
    imgs = [resize_safe(i) for i in imgs]
    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(i, (w, h)) for i in imgs]

    merge = cv2.createMergeMertens()
    imgs_f = [i.astype(np.float32)/255.0 for i in imgs]
    fused = merge.process(imgs_f)

    # CRITICAL FIX
    fused = np.nan_to_num(fused)
    fused = cv2.normalize(fused, None, 0, 1, cv2.NORM_MINMAX)

    out = np.clip(fused*255, 0, 255).astype(np.uint8)

    # final guard against black
    if out.mean() < 5:
        print("Fallback to base exposure (fusion too dark)")
        return imgs[1]

    return out

# ------------------------
# Worker loop
# ------------------------

def pick_job():
    jobs = sb_get("/rest/v1/hdr_jobs?select=*&status=eq.queued&limit=1")
    if isinstance(jobs, list) and jobs:
        return jobs[0]
    return None

def process_once():
    job = pick_job()
    if not job:
        return False

    job_id = job["id"]
    set_id = job["set_id"]

    print("Processing job:", job_id)

    sb_patch("hdr_jobs", job_id, {"status": "processing"})

    s = sb_get(f"/rest/v1/hdr_sets?id=eq.{set_id}&select=*")[0]

    f_under = sb_get(f"/rest/v1/hdr_files?id=eq.{s['file_under_id']}&select=*")[0]
    f_base  = sb_get(f"/rest/v1/hdr_files?id=eq.{s['file_base_id']}&select=*")[0]
    f_over  = sb_get(f"/rest/v1/hdr_files?id=eq.{s['file_over_id']}&select=*")[0]

    img_under = download_image(f_under["storage_path"])
    img_base  = download_image(f_base["storage_path"])
    img_over  = download_image(f_over["storage_path"])

    out = exposure_fusion([img_under, img_base, img_over])

    out_path = f"{s['order_id']}/{set_id}.jpg"
    upload_image(out_path, out)

    sb_patch("hdr_sets", set_id, {"status": "complete", "output_path": out_path})
    sb_patch("hdr_jobs", job_id, {"status": "complete"})

    print("Done:", set_id)
    return True

def main():
    while True:
        if not process_once():
            time.sleep(2)

if __name__ == "__main__":
    main()
