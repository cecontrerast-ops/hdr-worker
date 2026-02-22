import os
import time
import requests
import numpy as np
import cv2

# =========================
# CONFIG
# =========================
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

RAW_BUCKET = "hdr_raw"
OUT_BUCKET = "hdr_output"

POLL_SECONDS = 2


# =========================
# HELPERS
# =========================
def headers_json():
    return {
        "apikey": SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }


def headers_storage():
    return {
        "apikey": SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
    }


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rest_get(table: str, params: dict):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.get(url, headers=headers_json(), params=params, timeout=45)
    if r.status_code // 100 != 2:
        raise RuntimeError(f"REST GET {table} failed {r.status_code}: {r.text}")
    return r.json()


def rest_patch(table: str, where_params: dict, payload: dict):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.patch(url, headers=headers_json(), params=where_params, json=payload, timeout=45)
    if r.status_code // 100 != 2:
        raise RuntimeError(f"REST PATCH {table} failed {r.status_code}: {r.text}")


def pick_job():
    # Pick 1 queued job not locked
    rows = rest_get(
        "hdr_jobs",
        {
            "select": "id,set_id,status,attempts,locked_at",
            "status": "eq.queued",
            "locked_at": "is.null",
            "limit": "1",
        },
    )
    return rows[0] if rows else None


def get_set(set_id: str):
    rows = rest_get(
        "hdr_sets",
        {
            "select": "id,order_id,file_under_id,file_base_id,file_over_id,status,output_path",
            "id": f"eq.{set_id}",
            "limit": "1",
        },
    )
    if not rows:
        raise RuntimeError(f"hdr_sets not found for set_id={set_id}")
    return rows[0]


def get_file(file_id: str):
    rows = rest_get(
        "hdr_files",
        {
            "select": "id,order_id,filename,storage_path",
            "id": f"eq.{file_id}",
            "limit": "1",
        },
    )
    if not rows:
        raise RuntimeError(f"hdr_files not found for file_id={file_id}")
    return rows[0]


def normalize_storage_path(p: str) -> str:
    """
    Worker expects relative object path inside bucket.
    If someone stored 'hdr_raw/uuid/file.jpg' by mistake, strip leading 'hdr_raw/'.
    If someone stored '/hdr_raw/...' strip leading slash too.
    """
    p = p.strip().lstrip("/")
    if p.startswith(f"{RAW_BUCKET}/"):
        p = p[len(f"{RAW_BUCKET}/") :]
    return p


def storage_download(bucket: str, rel_path: str):
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{rel_path}"
    r = requests.get(url, headers=headers_storage(), timeout=90)
    if r.status_code // 100 != 2:
        raise RuntimeError(f"DOWNLOAD failed {r.status_code} for {bucket}/{rel_path}: {r.text}")

    buf = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image for {bucket}/{rel_path}")
    return img


def storage_upload_jpg(bucket: str, rel_path: str, bgr_img, quality: int = 92):
    ok, enc = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPG output")

    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{rel_path}"
    r = requests.post(
        url,
        headers={
            **headers_storage(),
            "Content-Type": "image/jpeg",
            "x-upsert": "true",
        },
        data=enc.tobytes(),
        timeout=90,
    )
    if r.status_code // 100 != 2:
        raise RuntimeError(f"UPLOAD failed {r.status_code} for {bucket}/{rel_path}: {r.text}")


def exposure_fusion(imgs_bgr):
    """
    Natural HDR (real-estate look) using OpenCV Mertens exposure fusion.
    """
    mertens = cv2.createMergeMertens(1.0, 1.0, 1.0)
    imgs_f = [i.astype(np.float32) / 255.0 for i in imgs_bgr]
    fused = mertens.process(imgs_f)
    out = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
    return out


def order_done(order_id: str) -> bool:
    rows = rest_get(
        "hdr_sets",
        {"select": "status", "order_id": f"eq.{order_id}"},
    )
    statuses = [r["status"] for r in rows]
    return bool(statuses) and all(s == "complete" for s in statuses)


# =========================
# MAIN WORK
# =========================
def process_once():
    print("Worker alive: checking for jobs...")

    job = pick_job()
    if not job:
        return False

    job_id = job["id"]
    set_id = job["set_id"]
    attempts = job.get("attempts", 0)

    print(f"Picked job {job_id} set_id={set_id} attempts={attempts}")

    try:
        # Lock job
        rest_patch("hdr_jobs", {"id": f"eq.{job_id}"}, {"status": "processing", "locked_at": now_iso(), "attempts": attempts + 1})

        s = get_set(set_id)
        order_id = s["order_id"]

        # Mark processing
        rest_patch("hdr_sets", {"id": f"eq.{set_id}"}, {"status": "processing"})
        rest_patch("hdr_orders", {"id": f"eq.{order_id}"}, {"status": "processing"})

        f_under = get_file(s["file_under_id"])
        f_base  = get_file(s["file_base_id"])
        f_over  = get_file(s["file_over_id"])

        p_under = normalize_storage_path(f_under["storage_path"])
        p_base  = normalize_storage_path(f_base["storage_path"])
        p_over  = normalize_storage_path(f_over["storage_path"])

        print("Paths:", p_under, p_base, p_over)

        imgs = [
            storage_download(RAW_BUCKET, p_under),
            storage_download(RAW_BUCKET, p_base),
            storage_download(RAW_BUCKET, p_over),
        ]

        out = exposure_fusion(imgs)

        out_path = f"{order_id}/{set_id}.jpg"
        print("Uploading output:", out_path)

        storage_upload_jpg(OUT_BUCKET, out_path, out)

        # Mark complete
        rest_patch("hdr_sets", {"id": f"eq.{set_id}"}, {"status": "complete", "output_path": out_path})
        rest_patch("hdr_jobs", {"id": f"eq.{job_id}"}, {"status": "complete"})

        if order_done(order_id):
            rest_patch("hdr_orders", {"id": f"eq.{order_id}"}, {"status": "complete"})

        print("Set complete:", set_id)
        return True

    except Exception as e:
        err = str(e)
        print("ERROR:", err)
        # Mark error
        try:
            rest_patch("hdr_jobs", {"id": f"eq.{job_id}"}, {"status": "error", "last_error": err})
        except Exception as e2:
            print("ERROR updating job:", str(e2))
        try:
            rest_patch("hdr_sets", {"id": f"eq.{set_id}"}, {"status": "error"})
        except Exception as e3:
            print("ERROR updating set:", str(e3))
        return True


def main():
    print("HDR Worker started.")
    while True:
        did = process_once()
        if not did:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
