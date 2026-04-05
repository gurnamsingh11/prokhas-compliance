"""
Thin wrapper: PDF path → extracted scheme JSON.
"""

import os
from dotenv import load_dotenv

from src.pdf_scheme_extractor.extract_schemes_from_pdf import PDFSchemeExtractor

load_dotenv()


def get_single_json(pdf_path: str) -> dict:
    """
    Extract all schemes from a PDF and return them as a dict.

    Uses OPENAI_API_KEY from the environment (or .env file).
    """
    extractor = PDFSchemeExtractor(api_key=os.getenv("api_key"))
    return extractor.extract_schemes_from_pdf(pdf_path)