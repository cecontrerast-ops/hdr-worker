import os
import time
import json
import traceback
from datetime import datetime, timezone

import requests
import numpy as np
import cv2


# =========================
# ENV
# =========================

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

RAW_BUCKET = os.getenv("RAW_BUCKET", "hdr_raw")
OUT_BUCKET = os.getenv("OUT_BUCKET", "hdr_output")

POLL_SECONDS = 2
MAX_ATTEMPTS = 6
JPEG_QUALITY = 92

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE env vars")

HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

def now_iso():
    return datetime.now(timezone.utc).isoformat()


# =========================
# SUPABASE HELPERS
# =========================

def sb_get(path):
    r = requests.get(f"{SUPABASE_URL}{path}", headers=HEADERS)
    r.raise_for_status()
    return r.json()

def sb_patch(path, payload):
    r = requests.patch(
        f"{SUPABASE_URL}{path}",
        headers=HEADERS,
        data=json.dumps(payload),
    )
    r.raise_for_status()
    return r.json()

def patch_row(table, row_id, payload):
    return sb_patch(f"/rest/v1/{table}?id=eq.{row_id}", payload)

def get_one(table, row_id):
    rows = sb_get(f"/rest/v1/{table}?id=eq.{row_id}&select=*")
    if not rows:
        return None
    return rows[0]


# =========================
# STORAGE
# =========================

def storage_download(bucket, path):
    r = requests.get(
        f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}",
        headers=HEADERS,
    )
    r.raise_for_status()
    return r.content

def storage_upload(bucket, path, img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")

    r = requests.put(
        f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}",
        headers={**HEADERS, "Content-Type": "image/jpeg"},
        data=buf.tobytes(),
    )
    r.raise_for_status()


# =========================
# IMAGE PROCESSING
# =========================

def align_to_base(img, base):
    try:
        im1 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
        cc, warp = cv2.findTransformECC(im1, im2, warp, cv2.MOTION_AFFINE, criteria)
        aligned = cv2.warpAffine(
            img,
            warp,
            (base.shape[1], base.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned
    except:
        return img


def exposure_fusion(under, base, over):
    merge = cv2.createMergeMertens()
    imgs = [
        under.astype(np.float32) / 255.0,
        base.astype(np.float32) / 255.0,
        over.astype(np.float32) / 255.0,
    ]
    fused = merge.process(imgs)
    fused = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
    return fused


def force_brightness(img, target_mean=140):
    """
    Hard brightness normalization so image NEVER appears black.
    """
    img = img.astype(np.float32)
    mean = img.mean()

    if mean < 1:
        mean = 1

    gain = target_mean / mean
    gain = np.clip(gain, 0.8, 6.0)

    img *= gain

    # soft highlight compression
    img = img / (1 + (img / 255.0) ** 2 * 0.4)

    return np.clip(img, 0, 255).astype(np.uint8)


def sharpen(img):
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1.6, blur, -0.6, 0)


def process_hdr(under, base, over):
    h, w = base.shape[:2]
    under = cv2.resize(under, (w, h))
    over = cv2.resize(over, (w, h))

    under = align_to_base(under, base)
    over = align_to_base(over, base)

    fused = exposure_fusion(under, base, over)

    # force visible brightness
    fused = force_brightness(fused, target_mean=150)

    fused = sharpen(fused)

    return fused


# =========================
# WORKER LOOP
# =========================

def pick_next_job():
    jobs = sb_get("/rest/v1/hdr_jobs?status=eq.queued&select=*&order=created_at.asc&limit=5")
    for j in jobs:
        if j.get("attempts", 0) >= MAX_ATTEMPTS:
            patch_row("hdr_jobs", j["id"], {"status": "error"})
            continue
        if j.get("locked_at"):
            continue
        patch_row("hdr_jobs", j["id"], {
            "status": "processing",
            "locked_at": now_iso(),
            "attempts": j.get("attempts", 0) + 1,
        })
        return j
    return None


def process_once():
    job = pick_next_job()
    if not job:
        print("Waiting...")
        return False

    job_id = job["id"]
    set_id = job["set_id"]

    try:
        s = get_one("hdr_sets", set_id)
        f_under = get_one("hdr_files", s["file_under_id"])
        f_base  = get_one("hdr_files", s["file_base_id"])
        f_over  = get_one("hdr_files", s["file_over_id"])

        img_under = cv2.imdecode(
            np.frombuffer(storage_download(RAW_BUCKET, f_under["storage_path"]), np.uint8),
            cv2.IMREAD_COLOR,
        )
        img_base = cv2.imdecode(
            np.frombuffer(storage_download(RAW_BUCKET, f_base["storage_path"]), np.uint8),
            cv2.IMREAD_COLOR,
        )
        img_over = cv2.imdecode(
            np.frombuffer(storage_download(RAW_BUCKET, f_over["storage_path"]), np.uint8),
            cv2.IMREAD_COLOR,
        )

        out = process_hdr(img_under, img_base, img_over)

        out_path = f"{s['order_id']}/{set_id}.jpg"
        storage_upload(OUT_BUCKET, out_path, out)

        patch_row("hdr_sets", set_id, {"status": "complete", "output_path": out_path})
        patch_row("hdr_jobs", job_id, {"status": "complete"})

        print("Done:", set_id)

    except Exception as e:
        print("ERROR:", str(e))
        patch_row("hdr_jobs", job_id, {"status": "error"})
        patch_row("hdr_sets", set_id, {"status": "error"})

    return True


def main():
    while True:
        did = process_once()
        if not did:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
