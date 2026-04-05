"""
PDF Scheme Extractor
====================
Extracts structured scheme data from a PDF by:
1. Rendering every page to a PNG image (preserves layout, handles tables).
2. Sending pages in batches to OpenAI's vision model.
3. Parsing the response with structured JSON output (strict schema).

All temp images are always deleted, even on failure.
"""

import os
import json
import base64
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pymupdf
from openai import OpenAI
from pydantic import BaseModel

# ────────────────────────── CONFIG ──────────────────────────── #

MODEL = "gpt-4o"          # supports vision + structured output
BATCH_SIZE = 15           # pages per LLM call
MAX_RETRIES = 3

logger = logging.getLogger(__name__)

# ────────────────────────── PYDANTIC SCHEMA ─────────────────── #
# Using Pydantic lets openai.beta.chat.completions.parse() give us
# a fully-validated, typed object back — no manual JSON.loads needed.

class Scheme(BaseModel):
    name_of_scheme: str
    objective: str
    total_allocation: str
    availability_period: str
    purpose_of_financing: str
    guarantee_coverage: str
    gurantee_fee: str
    payment_of_guarantee_fees: str
    eligibility: str
    customer_financial_record: str
    customer_credit_record_and_litigation_or_suit: str
    years_in_business: str
    type_of_facility: str
    financing_limit: str
    tenure_of_financing: str
    interest_or_profit_rate: str
    tangible_networth_TNW: str
    mandatory_security: str
    guideliness_for_application: str
    documents_required_to_be_submitted_upon_submitting_claims_to_SJPP: str
    BNM_features: str
    SJPP_right_to_audit: str
    other_terms: str


class SchemeList(BaseModel):
    schemes: List[Scheme]


# ────────────────────────── EXTRACTOR ───────────────────────── #

class PDFSchemeExtractor:
    """Thread-safe, stateless extractor. One instance can process many PDFs."""

    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    # ── Public ── #

    def extract_schemes_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main entry point.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            {"schemes": [<scheme dicts> ...]}
        """
        pdf_path = str(Path(pdf_path).resolve())
        logger.info("Processing PDF: %s", pdf_path)

        image_paths = self._render_pages(pdf_path)
        logger.info("Rendered %d page images", len(image_paths))

        try:
            batches = self._chunk(image_paths, BATCH_SIZE)
            all_schemes: List[Dict] = []

            for i, batch in enumerate(batches):
                logger.info("Batch %d / %d  (%d pages)", i + 1, len(batches), len(batch))
                result = self._process_batch_with_retry(batch)
                all_schemes.extend(result.get("schemes", []))

        finally:
            self._cleanup(image_paths)

        logger.info("Extracted %d scheme(s) total", len(all_schemes))
        return {"schemes": all_schemes}

    # ── Page rendering ── #

    def _render_pages(self, pdf_path: str) -> List[str]:
        """Render every PDF page to a temp PNG at 300 DPI."""
        doc = pymupdf.open(pdf_path)
        tmp_dir = tempfile.mkdtemp(prefix="pdf_scheme_")
        paths = []
        for idx in range(len(doc)):
            pix = doc[idx].get_pixmap(dpi=300)
            path = os.path.join(tmp_dir, f"page_{idx:04d}.png")
            pix.save(path)
            paths.append(path)
        return paths

    def _cleanup(self, image_paths: List[str]) -> None:
        deleted = 0
        dirs_to_remove = set()
        for path in image_paths:
            try:
                os.remove(path)
                deleted += 1
                dirs_to_remove.add(os.path.dirname(path))
            except OSError as exc:
                logger.warning("Could not delete %s: %s", path, exc)
        for d in dirs_to_remove:
            try:
                os.rmdir(d)
            except OSError:
                pass
        logger.info("Cleaned up %d temp file(s)", deleted)

    # ── OpenAI calls ── #

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    def _build_messages(self, batch: List[str]) -> List[Dict]:
        """
        Build the chat messages list for one batch.

        OpenAI multimodal format:
          content = [{"type": "text", ...}, {"type": "image_url", ...}, ...]
        """
        image_parts = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(p)}",
                    "detail": "high",
                },
            }
            for p in batch
        ]

        return [
            {
                "role": "system",
                "content": (
                    "You are a document extraction expert specialising in banking and finance. "
                    "Extract every scheme from the provided document images exactly as written. "
                    "Return empty string for any field that is not present — never omit a field."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL schemes from these document page images.\n\n"
                            "Rules:\n"
                            "- The document is tabular: left column = field name, right column = value.\n"
                            "- Each distinct scheme must be its own object.\n"
                            "- Preserve exact wording from the document.\n"
                            "- If a field is missing or not applicable, return an empty string (\"\").\n"
                            "- Do NOT skip partially visible tables or page-break-spanning tables.\n"
                            "- Do NOT return an empty list unless there is genuinely no scheme data."
                        ),
                    },
                    *image_parts,
                ],
            },
        ]

    def _process_batch_with_retry(self, batch: List[str]) -> Dict:
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._process_batch(batch)
            except Exception as exc:
                last_exc = exc
                logger.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
        logger.error("All %d retries exhausted. Last error: %s", MAX_RETRIES, last_exc)
        return {"schemes": []}

    def _process_batch(self, batch: List[str]) -> Dict:
        """
        Call OpenAI with structured output (Pydantic model).
        Uses the beta parse helper which validates and returns a typed object.
        """
        messages = self._build_messages(batch)

        completion = self.client.beta.chat.completions.parse(
            model=MODEL,
            messages=messages,
            response_format=SchemeList,
            temperature=0,  # deterministic extraction
        )

        choice = completion.choices[0]

        # Refusal guard
        if choice.finish_reason == "refusal":
            logger.warning("Model refused to process batch.")
            return {"schemes": []}

        parsed: SchemeList = choice.message.parsed
        return parsed.model_dump()

    # ── Utility ── #

    @staticmethod
    def _chunk(lst: List, size: int) -> List[List]:
        return [lst[i : i + size] for i in range(0, len(lst), size)]