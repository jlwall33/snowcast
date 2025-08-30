from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os, io
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import storage

app = FastAPI(title="Snowcast API")

@app.get("/")
def root():
    return {"ok": True, "service": "snowcast-api"}

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "bucket": os.getenv("BUCKET", ""),
        "project": os.getenv("PROJECT_ID", ""),
        "region": os.getenv("REGION", "")
    }

class RenderRequest(BaseModel):
    run_id: str | None = None
    valid_time: str | None = None
    bbox: list[float] | None = None
    snow_ratio: int | None = 10

def _upload_bytes_to_gcs(data: bytes, bucket: str, path: str) -> str:
    client = storage.Client()
    b = client.bucket(bucket)
    blob = b.blob(path)
    blob.upload_from_string(data, content_type="image/png")
    return f"gs://{bucket}/{path}"

@app.post("/render")
def render_png(req: RenderRequest):
    try:
        bucket = os.getenv("BUCKET", "")
        if not bucket:
            return {"ok": False, "error": "Missing BUCKET env"}

        run_id = req.run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        valid = req.valid_time or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        ratio = req.snow_ratio or 10

        # demo image
        arr = np.random.rand(180, 240) * 12.0

        fig, ax = plt.subplots(figsize=(6,4), dpi=150)
        im = ax.imshow(arr, origin="lower")
        ax.set_title(f"Snowfall (in) — {valid}")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Snowfall (in)")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        data = buf.getvalue()

        out = f"products/{run_id}/png/{valid.replace(':','').replace('-','').replace('T','_').replace('Z','')}.png"
        gs_uri = _upload_bytes_to_gcs(data, bucket, out)
        public_url = f"https://storage.googleapis.com/{bucket}/{out}"

        return {"ok": True, "gs_uri": gs_uri, "public_url": public_url,
                "bytes": len(data), "units": "in", "snow_ratio": ratio}

    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
