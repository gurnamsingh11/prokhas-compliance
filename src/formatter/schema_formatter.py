"""
Schema formatter
================
Converts two JSON blobs (before / after) into a flat list of
(field_name, before_value, after_value) triples ready for comparison.

The top-level structure from the extractor is:
    {"schemes": [<scheme_obj>, <scheme_obj>, ...]}

Each scheme_obj is a flat dict of string fields.

Comparison strategy
-------------------
* Schemes are paired by position (scheme 0 before ↔ scheme 0 after).
* If the counts differ, the shorter side is padded with empty dicts so
  every field still gets a row (with an empty "after" or "before" value).
* Field names are prefixed with the scheme index so the caller always
  knows which scheme a difference belongs to.
"""

import json
from typing import Any, List, Tuple


def format_value(val: Any) -> str:
    """Serialise any value to a human-readable string."""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False, indent=2)
    if val is None:
        return ""
    return str(val)


def compare_json_to_list(
    before: dict,
    after: dict,
) -> List[Tuple[str, str, str]]:
    """
    Flatten two {"schemes": [...]} dicts into a list of comparison triples.

    Args:
        before: Extracted JSON for the "before" PDF.
        after:  Extracted JSON for the "after" PDF.

    Returns:
        List of (qualified_field_name, before_value, after_value) tuples.
        qualified_field_name format:  "scheme_<N>.<field>"
        e.g. "scheme_0.financing_limit"
    """
    schemes_before: List[dict] = before.get("schemes", [])
    schemes_after: List[dict] = after.get("schemes", [])

    # Pad the shorter list so we never silently drop schemes
    max_len = max(len(schemes_before), len(schemes_after))
    schemes_before = schemes_before + [{}] * (max_len - len(schemes_before))
    schemes_after = schemes_after + [{}] * (max_len - len(schemes_after))

    result: List[Tuple[str, str, str]] = []

    for idx, (sb, sa) in enumerate(zip(schemes_before, schemes_after)):
        # Union of keys from both sides so nothing is silently ignored
        all_keys = list(sb.keys()) + [k for k in sa if k not in sb]

        for key in all_keys:
            result.append(
                (
                    key,
                    format_value(sb.get(key, "")),
                    format_value(sa.get(key, "")),
                )
            )

    return result


def format_json(data):
    key_mapping = {
        "name_of_scheme": "Name of Scheme",
        "objective": "Objective",
        "total_allocation": "Total Allocation",
        "availability_period": "Availability Period",
        "purpose_of_financing": "Purpose of Financing",
        "guarantee_coverage": "Guarantee Coverage",
        "gurantee_fee": "Guarantee Fee",
        "payment_of_guarantee_fees": "Payment of Guarantee Fees",
        "eligibility": "Eligibility",
        "customer_financial_record": "Customer's Financial Record",
        "customer_credit_record_and_litigation_or_suit": "Customer's credit records and litigation / suit",
        "years_in_business": "Years in Business",
        "type_of_facility": "Type of Facility",
        "financing_limit": "Financing Limit",
        "tenure_of_financing": "Tenure of Financing",
        "interest_or_profit_rate": "Interest / Profit Rate",
        "tangible_networth_TNW": 'Tangible Networth ("TNW")',
        "mandatory_security": "Mandatory Security",
        "guideliness_for_application": "Guideliness for Application",
        "documents_required_to_be_submitted_upon_submitting_claims_to_SJPP": "Documents required to be submitted upon submitting claims to SJPP",
        "BNM_features": "BNM Features",
        "SJPP_right_to_audit": "SJPP's Right to Audit",
        "other_terms": "Other Terms"
    }

    data["results"] = {
    key_mapping.get(key, key): value
    for key, value in data["results"].items()
    }
    return data