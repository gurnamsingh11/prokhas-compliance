"""
PDF Scheme Comparator — FastAPI entry point
===========================================
Usage:
    uvicorn main:app --reload

POST /compare
    Form fields:
        before  (UploadFile) — the original PDF
        after   (UploadFile) — the updated PDF

Returns:
    {
        "results": {
            "financing_limit": {"difference": true,  "reason": "..."},
            "eligibility":     {"difference": false, "reason": "..."},
            ...
        },
        "summary": {
            "total_fields": N,
            "fields_with_differences": M
        }
    }
"""

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.difference.main import main as compare_pdfs
from src.formatter.schema_formatter import format_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Scheme Comparator",
    description="Extracts and semantically compares scheme fields from two PDFs.",
    version="1.0.0",
)


async def _save_upload(upload: UploadFile, suffix: str, tmp_dir: str) -> str:
    """Write an UploadFile to a temp path and return the path."""
    dest = Path(tmp_dir) / f"{upload.filename or 'file'}{suffix}"
    contents = await upload.read()
    dest.write_bytes(contents)
    return str(dest)


@app.post("/compare", response_class=JSONResponse)
async def compare(
    before: UploadFile = File(..., description="Original PDF"),
    after: UploadFile = File(..., description="Updated PDF"),
):
    for upload, label in ((before, "before"), (after, "after")):
        if upload.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(
                status_code=415,
                detail=f"'{label}' must be a PDF (got {upload.content_type!r}).",
            )

    with tempfile.TemporaryDirectory(prefix="scheme_cmp_") as tmp_dir:
        before_path = await _save_upload(before, "_before.pdf", tmp_dir)
        after_path  = await _save_upload(after,  "_after.pdf",  tmp_dir)

        logger.info("Comparing  before=%s  after=%s", before_path, after_path)
        try:
            result = format_json(compare_pdfs(before_path, after_path))
        except Exception as exc:
            logger.exception("Comparison failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result