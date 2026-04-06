"""
Difference engine
=================
1. Extracts structured JSON from two PDFs (before / after).
2. Flattens both into a list of field triples.
3. For each field, calls an LLM to decide if there is a meaningful
   semantic difference and why.
4. Returns a single aggregated JSON dict.

LangChain structured-output approach
-------------------------------------
`model.with_structured_output(schema)` is the correct, stable LangChain
API for constrained generation.  The old `create_agent` + `ToolStrategy`
combo does not exist in any public LangChain release.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from src.difference.get_json import get_single_json
from src.formatter.schema_formatter import compare_json_to_list
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

MAX_WORKERS: int = int(os.getenv("COMPARE_MAX_WORKERS", "10"))


from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
# from src.llm_gateway.llm_model import get_chat_model 
from src.llm_gateway.llm_model import get_openai_model

# model = get_chat_model()
model = get_openai_model()

# JSON Schema (this is the key part)
difference_schema = {
    "type": "object",
    "description": "Semantic comparison between before and after values of a field",
    "properties": {
        "difference": {
            "type": "boolean",
            "description": "Whether there is a meaningful semantic difference"
        },
        "reason": {
            "type": "string",
            "description": "Explanation of the semantic difference or similarity"
        },
        "before": {
            "type": "string",
            "description": "The before value of the field"
        },
        "after": {
            "type": "string",
            "description": "The after value of the field"
        }
    },
    "required": ["difference", "reason"]
}

# Create agent with structured output
agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(difference_schema),
    system_prompt=(
        "You are a banking domain expert.\n"
        "Compare BEFORE and AFTER values of a given field.\n"
        "Return TRUE only if there is a meaningful semantic difference.\n"
        "Ignore minor wording changes, formatting, or rephrasing.\n"
        "Focus on meaning, financial implications, eligibility changes, limits, etc."
    )
)

# Function to call model
def compare_values(field_name: str, before: str, after: str):
    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
Field: {field_name}

BEFORE:
{before}

AFTER:
{after}
"""
            }
        ]
    })

    return response["structured_response"]


# ── Main orchestrator ────────────────────────────────────────── #

def main(pdf_before: str, pdf_after: str) -> Dict[str, Any]:
    """
    Full pipeline: extract → flatten → compare → aggregate.
 
    Args:
        pdf_before: Path to the original PDF.
        pdf_after:  Path to the updated PDF.
 
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
    logger.info("Extracting JSON from BEFORE PDF: %s", pdf_before)
    json_before = get_single_json(pdf_before)
 
    logger.info("Extracting JSON from AFTER PDF: %s", pdf_after)
    json_after = get_single_json(pdf_after)
 
    triples: List[Tuple[str, str, str]] = compare_json_to_list(
        before=json_before, after=json_after
    )
    logger.info("Comparing %d field(s)…", len(triples))
 
    # Pre-allocate a list so results slot in at the right index
    # regardless of which future finishes first.
    comparisons: List[Optional[Dict[str, Any]]] = [None] * len(triples)
 
    def _compare_indexed(idx: int, field_name: str, before: str, after: str):
        return idx, compare_values(field_name, before, after)
 
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_compare_indexed, i, fn, bv, av): i
            for i, (fn, bv, av) in enumerate(triples)
        }
        for future in as_completed(futures):
            idx, comparison = future.result()
            comparisons[idx] = comparison
            logger.debug(
                "  [%s] difference=%s | %s",
                triples[idx][0],
                comparison.get("difference"),
                comparison.get("reason", "")[:80],
            )
 
    # Build ordered results dict — order matches the original triples list.
    results: Dict[str, Any] = {
        field_name: comparisons[i]
        for i, (field_name, _, _) in enumerate(triples)
    }
    fields_with_differences = sum(
        1 for v in results.values() if v and v.get("difference")
    )
 
    output = {
        "results": results,
        "summary": {
            "total_fields": len(triples),
            "fields_with_differences": fields_with_differences,
        },
    }
 
    logger.info(
        "Done. %d / %d field(s) differ.",
        fields_with_differences,
        len(triples),
    )
    return output