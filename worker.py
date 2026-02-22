import os, time, requests
import numpy as np
import cv2

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
RAW_BUCKET = "hdr_raw"
OUT_BUCKET = "hdr_output"

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

def pick_job():
    # Pick 1 queued job with no lock
    url = f"{SUPABASE_URL}/rest/v1/hdr_jobs"
    params = {
        "select": "id,set_id,status,attempts",
        "status": "eq.queued",
        "locked_at": "is.null",
        "limit": "1",
    }
    r = requests.get(url, headers=headers_json(), params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()
    return rows[0] if rows else None

def patch_row(table, row_id, payload):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    params = {"id": f"eq.{row_id}"}
    r = requests.patch(url, headers=headers_json(), params=params, json=payload, timeout=30)
    r.raise_for_status()

def get_one(table, select_cols, row_id):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    params = {"select": select_cols, "id": f"eq.{row_id}"}
    r = requests.get(url, headers=headers_json(), params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise RuntimeError(f"{table} not found for id={row_id}")
    return rows[0]

def get_set(set_id):
    return get_one("hdr_sets", "id,order_id,file_under_id,file_base_id,file_over_id,status", set_id)

def get_file(file_id):
    return get_one("hdr_files", "id,storage_path,filename,order_id", file_id)

def storage_download(bucket, path):
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    r = requests.get(url, headers=headers_storage(), timeout=90)
    r.raise_for_status()
    buf = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode {bucket}/{path}")
    return img

def storage_upload_jpg(bucket, path, bgr, quality=92):
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode jpg")
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    r = requests.post(
        url,
        headers={**headers_storage(), "Content-Type": "image/jpeg", "x-upsert": "true"},
        data=enc.tobytes(),
        timeout=90,
    )
    if r.status_code // 100 != 2:
        raise RuntimeError(f"Upload failed {r.status_code}: {r.text}")

def exposure_fusion(imgs_bgr):
    # Natural HDR: Mertens exposure fusion
    mertens = cv2.createMergeMertens(1.0, 1.0, 1.0)
    imgs_f = [i.astype(np.float32) / 255.0 for i in imgs_bgr]
    fused = mertens.process(imgs_f)
    out = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
    return out

def order_done(order_id):
    url = f"{SUPABASE_URL}/rest/v1/hdr_sets"
    params = {"select": "status", "order_id": f"eq.{order_id}"}
    r = requests.get(url, headers=headers_json(), params=params, timeout=30)
    r.raise_for_status()
    statuses = [x["status"] for x in r.json()]
    return bool(statuses) and all(s == "complete" for s in statuses)

def process_once():
    job = pick_job()
    if not job:
        return False
    job_id = job["id"]
    set_id = job["set_id"]
    attempts = job["attempts"]
    try:
        # Lock job + mark processing
        patch_row("hdr_jobs", job_id, {"status": "processing", "locked_at": now_iso(), "attempts": attempts + 1})
        s = get_set(set_id)
        order_id = s["order_id"]
        patch_row("hdr_sets", set_id, {"status": "processing"})
        patch_row("hdr_orders", order_id, {"status": "processing"})
        f_under = get_file(s["file_under_id"])
        f_base  = get_file(s["file_base_id"])
        f_over  = get_file(s["file_over_id"])
        imgs = [
            storage_download(RAW_BUCKET, f_under["storage_path"]),
            storage_download(RAW_BUCKET, f_base["storage_path"]),
            storage_download(RAW_BUCKET, f_over["storage_path"]),
        ]
        out = exposure_fusion(imgs)
        out_path = f"{order_id}/{set_id}.jpg"  # relative path in hdr_output bucket
        storage_upload_jpg(OUT_BUCKET, out_path, out)
        patch_row("hdr_sets", set_id, {"status": "complete", "output_path": out_path})
        patch_row("hdr_jobs", job_id, {"status": "complete"})
        if order_done(order_id):
            patch_row("hdr_orders", order_id, {"status": "complete"})
        return True
    except Exception as e:
        patch_row("hdr_jobs", job_id, {"status": "error", "last_error": str(e)})
        try:
            patch_row("hdr_sets", set_id, {"status": "error"})
        except Exception:
            pass
        return True

def main():
    while True:
        did = process_once()
        if not did:
            time.sleep(2)

if __name__ == "__main__":
    main()

