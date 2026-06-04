from __future__ import annotations

SUMMARY_PROMPT = """You are summarizing CURRENT_ATTEMPT into compact route memory for an iterative search.

You are given:
- PROBLEM: the original problem being solved
- PRIOR_SUMMARIES: summaries of earlier attempts, possibly empty
- CURRENT_ATTEMPT: the full reasoning trace to summarize

Use PRIOR_SUMMARIES only to record how CURRENT_ATTEMPT relates to already-tried routes.

Return exactly one JSON object with these keys:
- route: 50-120 words summarizing what CURRENT_ATTEMPT actually tried, including the main path and any major abandoned branch
- route_signature: 3-6 short tags of the form "dimension: current choice"
- relationship_to_prior: an object with:
  - differences: 0-5 short tags naming concrete ways CURRENT_ATTEMPT differs from the most relevant PRIOR_SUMMARIES
  - overlap: 0-3 short tags naming important choices CURRENT_ATTEMPT reused from PRIOR_SUMMARIES

Guidance:
- Summarize only CURRENT_ATTEMPT in route and route_signature.
- Use PRIOR_SUMMARIES only for relationship_to_prior.
- If PRIOR_SUMMARIES is empty, use empty arrays for differences and overlap.
- Capture actual route differences and overlap; do not invent novelty.
- Do not include the attempt's final answer, option label, boxed value, or chosen conclusion anywhere in the output.
- Do not critique correctness or suggest fixes, lemmas, experiments, or next steps.
- Good signatures identify concrete framings, representations, cases, invariants, examples, tools, algorithms, hypotheses, or bottlenecks.
- Keep each route_signature, differences, and overlap tag short.
"""

SUMMARY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "maxLength": 900},
        "route_signature": {
            "type": "array",
            "items": {"type": "string", "maxLength": 64},
            "minItems": 3,
            "maxItems": 6,
        },
        "relationship_to_prior": {
            "type": "object",
            "properties": {
                "differences": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 64},
                    "minItems": 0,
                    "maxItems": 5,
                },
                "overlap": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 64},
                    "minItems": 0,
                    "maxItems": 3,
                },
            },
            "required": ["differences", "overlap"],
            "additionalProperties": False,
        },
    },
    "required": [
        "route",
        "route_signature",
        "relationship_to_prior",
    ],
    "additionalProperties": False,
}
