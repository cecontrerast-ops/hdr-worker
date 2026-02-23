import os
import time
import json
import base64
import traceback
from datetime import datetime, timezone

import requests
import numpy as np
import cv2


# =========================
# ENV / CONFIG
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
RAW_BUCKET = os.getenv("RAW_BUCKET", "hdr_raw")
OUT_BUCKET = os.getenv("OUT_BUCKET", "hdr_output")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "2"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "6"))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars")

HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
    # IMPORTANT: tells PostgREST to return JSON for PATCH/POST if possible
    "Prefer": "return=representation",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# =========================
# SUPABASE REST HELPERS
# =========================
def sb_get(path):
    url = f"{SUPABASE_URL}{path}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    if r.text.strip():
        return r.json()
    return []


def sb_patch(path, body):
    """
    PATCH sometimes returns 204 No Content. Never blindly call r.json().
    """
    url = f"{SUPABASE_URL}{path}"
    r = requests.patch(url, headers=HEADERS, data=json.dumps(body), timeout=30)
    r.raise_for_status()
    if r.text.strip():
        return r.json()
    return {}


def sb_post(path, body):
    """
    POST sometimes returns empty body if Prefer is ignored. Handle safely.
    """
    url = f"{SUPABASE_URL}{path}"
    r = requests.post(url, headers=HEADERS, data=json.dumps(body), timeout=30)
    r.raise_for_status()
    if r.text.strip():
        return r.json()
    return {}


# =========================
# STORAGE HELPERS
# =========================
def storage_download(bucket, path):
    """
    Uses Supabase Storage download endpoint.
    Returns bytes.
    """
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.content


def storage_upload_jpg(bucket, path, img_bgr):
    """
    Uploads JPG bytes to Supabase Storage.
    """
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode JPG")

    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    r = requests.post(
        url,
        headers={**HEADERS, "Content-Type": "image/jpeg"},
        data=enc.tobytes(),
        timeout=120,
    )
    # If file exists, try upsert with PUT:
    if r.status_code in (409, 400):
        r = requests.put(
            url,
            headers={**HEADERS, "Content-Type": "image/jpeg"},
            data=enc.tobytes(),
            timeout=120,
        )
    r.raise_for_status()
    return True


# =========================
# IMAGE OPS
# =========================
def decode_jpg_to_bgr(jpg_bytes):
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Could not decode image")
    return img


def safe_downscale(img, max_dim=2000):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def exposure_fusion(imgs_bgr):
    """
    SAFE Mertens exposure fusion:
    - downscale before fusion (prevents OOM)
    - enforce same size
    """
    try:
        if not imgs_bgr or len(imgs_bgr) < 3:
            raise RuntimeError("Need 3 images")

        # downscale each first
        resized = [safe_downscale(im, 2000) for im in imgs_bgr]

        # enforce same size as first
        h, w = resized[0].shape[:2]
        resized = [cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) for im in resized]

        mertens = cv2.createMergeMertens()
        imgs_f = [im.astype(np.float32) / 255.0 for im in resized]
        fused = mertens.process(imgs_f)  # float 0..1

        out = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
        return out

    except Exception as e:
        print("FUSION ERROR:", str(e))
        # fallback: middle exposure
        return imgs_bgr[1]


# =========================
# DB HELPERS
# =========================
def get_set(set_id):
    rows = sb_get(f"/rest/v1/hdr_sets?id=eq.{set_id}&select=*")
    return rows[0] if rows else None


def get_file(file_id):
    rows = sb_get(f"/rest/v1/hdr_files?id=eq.{file_id}&select=*")
    return rows[0] if rows else None


def order_done(order_id):
    rows = sb_get(f"/rest/v1/hdr_sets?order_id=eq.{order_id}&select=status")
    if not rows:
        return False
    return all(r.get("status") == "complete" for r in rows)


def patch_row(table, row_id, payload):
    # returns representation if Prefer honored, else {}
    return sb_patch(f"/rest/v1/{table}?id=eq.{row_id}", payload)


# =========================
# JOB LOOP
# =========================
def pick_job():
    """
    Find one queued, unlocked job.
    """
    rows = sb_get(
        "/rest/v1/hdr_jobs"
        "?select=*"
        "&status=eq.queued"
        "&locked_at=is.null"
        "&limit=1"
    )
    return rows[0] if rows else None


def process_once():
    job = pick_job()
    if not job:
        print("Worker alive: checking for jobs...")
        return False

    job_id = job["id"]
    set_id = job["set_id"]
    attempts = int(job.get("attempts") or 0)

    print(f"Picked job {job_id} set_id={set_id} attempts={attempts}")

    # lock job
    lock_time = now_iso()
    patch_row("hdr_jobs", job_id, {"locked_at": lock_time, "status": "processing", "attempts": attempts + 1})

    s = get_set(set_id)
    if not s:
        patch_row("hdr_jobs", job_id, {"status": "error", "last_error": "Set not found"})
        return True

    order_id = s["order_id"]
    patch_row("hdr_sets", set_id, {"status": "processing"})
    patch_row("hdr_orders", order_id, {"status": "processing"})

    try:
        f_under = get_file(s["file_under_id"])
        f_base = get_file(s["file_base_id"])
        f_over = get_file(s["file_over_id"])

        if not f_under or not f_base or not f_over:
            raise RuntimeError("Missing file rows for bracket")

        paths = [f_under["storage_path"], f_base["storage_path"], f_over["storage_path"]]
        print("Paths:", *paths)

        imgs = []
        for p in paths:
            print(f"Downloading: {RAW_BUCKET}/{p}")
            data = storage_download(RAW_BUCKET, p)
            imgs.append(decode_jpg_to_bgr(data))

        out = exposure_fusion(imgs)

        out_path = f"{order_id}/{set_id}.jpg"
        print(f"Uploading: {OUT_BUCKET}/{out_path}")
        storage_upload_jpg(OUT_BUCKET, out_path, out)

        patch_row("hdr_sets", set_id, {"status": "complete", "output_path": out_path})
        patch_row("hdr_jobs", job_id, {"status": "complete", "locked_at": None})

        if order_done(order_id):
            patch_row("hdr_orders", order_id, {"status": "complete"})

        print(f"Completed set {set_id}")
        return True

    except Exception as e:
        err = f"{str(e)}\n{traceback.format_exc()}"
        print("ERROR:", err)

        patch_row("hdr_jobs", job_id, {"status": "error", "last_error": str(e), "locked_at": None})
        try:
            patch_row("hdr_sets", set_id, {"status": "error"})
        except Exception:
            pass

        # retry logic: re-queue if attempts left
        if attempts + 1 < MAX_ATTEMPTS:
            patch_row("hdr_jobs", job_id, {"status": "queued", "locked_at": None})
        return True


def main():
    print("HDR Worker started.")
    while True:
        did = process_once()
        if not did:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
