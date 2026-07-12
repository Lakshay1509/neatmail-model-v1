import os
import re
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import uuid

load_dotenv()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    expected_api_key = os.environ.get("DASHBOARD_API_KEY")
    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key not configured on the server"
        )
    if api_key_header == expected_api_key:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key"
    )

app = FastAPI(dependencies=[Security(get_api_key)])

# Make sure to set your OPENAI_API_KEY environment variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "neatmail-corrections"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
SIMILARITY_THRESHOLD = 0.82

class Tag(BaseModel):
    name: str
    description: Optional[str] = None
    user_defined: bool = False

class EmailRequest(BaseModel):
    user_id: str
    subject: str
    from_: str = Field(alias="from")
    bodySnippet: str
    tags: List[Tag]
    sensitivity: str

class CorrectionRequest(BaseModel):
    user_id: str
    subject: str
    body: str
    correct_label: str
    wrong_label: Optional[str] = None

class DeleteUserRequest(BaseModel):
    user_id: str

class EmailClassificationResult(BaseModel):
    category: str
    response_required: bool
    ai_summary: str = ""
    ai_action: str = ""


CLASSIFY_SYSTEM_PROMPT = """You are an email classifier. Return ONLY valid JSON.

OVERRIDE RULE: User corrections in <user_corrections> take precedence over all general rules below. If the email closely resembles a correction, apply that label.

CATEGORY PRIORITY: When choosing a category from <categories>, user-defined tags (marked with "[user-defined]") take precedence over system tags. First try to classify into a user-defined tag. Only if no user-defined tag fits should you fall back to a system tag.

Automated sender (noreply/notifications/billing/alerts/mailer/digest/newsletter, or any platform like GitHub/Slack/Stripe/Google, or template body with no personal reply expected):
  \u2192 response_required=false, never "Pending Response", ai_summary+ai_action=""

category \u2014 pick exactly one from the provided list. MUST return "" if confidence <95%.
  "Pending Response" \u2192 human only. ALWAYS populate ai_summary+ai_action.
  "Action Needed" \u2192 populate ai_summary+ai_action ONLY when email needs human judgment: approvals, contracts, meeting confirmations, expiring subscription, account suspended, invoice due.
    Leave "" for one-click triggers: verification, OTP, password reset, 2FA, order confirmations, shipping notifications.
  All other categories (newsletters, alerts, marketing, read-only) \u2192 ai_summary+ai_action="" ALWAYS.

response_required \u2014 false for automated senders. true ONLY when a human sender explicitly expects a reply.

ai_summary \u2014 12-15 words, active voice. Human emails \u2192 urgent, lead with risk/ask. Automated critical \u2192 calm, state the decision needed.

ai_action \u2014 2-3 words, imperative verb-first from: Escalate now|Reply with ETA|Review & approve|Send feedback|Confirm availability|Approve invoices|Read later|Review billing|Check activity|Submit proposal|Renew or review|Investigate now|Reconnect now
"""

DIGEST_SENDER_RE = re.compile(r'digest@send\.neatmail\.app', re.IGNORECASE)


def _is_digest_sender(from_: str) -> bool:
    return bool(DIGEST_SENDER_RE.search(from_))


def _normalize_tag(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


def match_category(parsed_category: str, tags: List[Tag]) -> str:
    """Match parsed category against available tags. Returns matched tag name or empty string."""
    if not parsed_category:
        return ""
    normalized_parsed = _normalize_tag(parsed_category)

    # 1. Try exact normalized match
    for t in tags:
        if _normalize_tag(t.name) == normalized_parsed:
            return t.name

    # 2. Fallback: substring matching
    if len(normalized_parsed) > 2:
        for t in tags:
            nt = _normalize_tag(t.name)
            if nt in normalized_parsed or normalized_parsed in nt:
                return t.name

    return ""


def is_actionable_category(category: str) -> bool:
    # "Pending Response" is only valid for human senders — the prompt hard-blocks it
    # for automated senders, so by the time we reach here it is always a human email.
    return _normalize_tag(category) in {"actionneeded", "pendingresponse"}


# ── Batch endpoint models ──────────────────────────────────────────────

class MaxBatchSizeExceededError(Exception):
    pass


class BatchEmailItem(BaseModel):
    id: str
    user_id: str
    bodySnippet: str
    subject: str
    from_: str = Field(alias="from")
    tags: List[Tag]
    sensitivity: str = "if actionable"


class BatchClassifyRequest(BaseModel):
    requests: List[BatchEmailItem]


class BatchEmailResult(BaseModel):
    id: str
    category: str
    response_required: bool
    ai_summary: str = ""
    ai_action: str = ""


class BatchClassifyResponse(BaseModel):
    results: List[BatchEmailResult]


MAX_BATCH_SIZE = 10


def init_index():
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

index = init_index()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def make_snippet(subject: str, body: str) -> str:
    return f"{subject}\n{body[:200]}"


def save_correction(
    user_id: str,
    subject: str,
    body: str,
    correct_label: str,
    wrong_label: str = None
):
    snippet = make_snippet(subject, body)
    embedding = embed(snippet)

    index.upsert(vectors=[{
        "id": str(uuid.uuid4()),
        "values": embedding,
        "metadata": {
            "user_id": user_id,
            "snippet": snippet,
            "correct_label": correct_label,
            "wrong_label": wrong_label or "",
        }
    }])
    print(f"✓ Correction saved: '{wrong_label}' → '{correct_label}'")

def get_corrections(user_id: str, subject: str, body: str, top_k: int = 3) -> list[dict]:
    snippet = make_snippet(subject, body)
    embedding = embed(snippet)

    results = index.query(
        vector=embedding,
        top_k=top_k,
        filter={"user_id": {"$eq": user_id}},  # scoped per user
        include_metadata=True
    )

    corrections = []
    for match in results.matches:
        if match.score >= SIMILARITY_THRESHOLD:
            corrections.append({
                "snippet": match.metadata["snippet"],
                "correct_label": match.metadata["correct_label"],
                "wrong_label": match.metadata["wrong_label"],
                "score": match.score
            })

    return corrections

def sanitize_prompt_text(s: str) -> str:
    """Replace double quotes with single quotes to avoid breaking prompt quoting."""
    return s.replace('"', "'")


def build_few_shot_block(corrections: list[dict]) -> str:
    if not corrections:
        return ""

    lines = ["<user_corrections>"]
    lines.append("OVERRIDE RULE: If the current email closely resembles any correction below, the correction's label takes precedence over general rules.")
    for i, c in enumerate(corrections, 1):
        wrong_label = sanitize_prompt_text(c["wrong_label"])
        correct_label = sanitize_prompt_text(c["correct_label"])
        snippet = sanitize_prompt_text(c["snippet"][:100])
        wrong = f" (was wrongly labelled: {wrong_label})" if wrong_label else ""
        lines.append(f"  [{i}] Email snippet: \"{snippet}...\"")
        lines.append(f"       Correct label: {correct_label}{wrong}")
    lines.append("</user_corrections>")

    return "\n".join(lines)


def classify_email(email_data: EmailRequest) -> EmailClassificationResult:
    corrections = get_corrections(email_data.user_id, email_data.subject, email_data.bodySnippet)
    few_shot_block = build_few_shot_block(corrections)

    tags = email_data.tags
    tag_context_lines = []
    for t in tags:
        prefix = "[user-defined] " if t.user_defined else ""
        desc = t.description.strip() if t.description and t.description.strip() else "No description provided"
        tag_context_lines.append(f"- {prefix}{t.name}: {desc}")
    tag_context = "\n".join(tag_context_lines)

    sensitivity_guidance = _get_sensitivity_guidance(email_data.sensitivity)

    user_prompt = f"""<email>
Subject: {email_data.subject}
From: {email_data.from_}
Body: {email_data.bodySnippet}
</email>

{few_shot_block}

<categories>
{tag_context}
</categories>

<sensitivity_rule>
{sensitivity_guidance}
</sensitivity_rule>

Classify. Return valid JSON."""

    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "email_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "response_required": {"type": "boolean"},
                    "ai_summary": {"type": "string"},
                    "ai_action": {"type": "string"}
                },
                "required": ["category", "response_required", "ai_summary", "ai_action"],
                "additionalProperties": False
            }
        }
    }

    try:
       completion = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        response_format=schema,
        reasoning_effort="medium",
        max_completion_tokens=6000,
        seed=42,
    )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    content = completion.choices[0].message.content
    if not content:
        finish_reason = completion.choices[0].finish_reason
        reasoning_tokens = getattr(completion.usage.completion_tokens_details, "reasoning_tokens", None) if completion.usage else None
        raise HTTPException(
            status_code=500,
            detail=f"No response from OpenAI (finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens}, max_completion_tokens=6000)"
        )

    try:
        parsed_json = json.loads(content)
        parsed_category = parsed_json.get("category", "")

        category = match_category(parsed_category, tags)
        is_actionable = is_actionable_category(category)

        ai_summary = parsed_json.get("ai_summary", "") if is_actionable else ""
        ai_action = parsed_json.get("ai_action", "") if is_actionable else ""

        if _is_digest_sender(email_data.from_):
            ai_summary = ""
            ai_action = ""

        return EmailClassificationResult(
            category=category,
            response_required=parsed_json.get("response_required", False),
            ai_summary=ai_summary,
            ai_action=ai_action
        )
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from OpenAI")


SENSITIVITY_MAP = {
    "always draft": "Treat response_required as true for nearly all human-sent emails; false only for obvious automated/no-reply messages.",
    "if known sender AND directly addressed": "Set response_required=true only if the sender seems personally known and the email directly asks this user to respond.",
    "if actionable": "Set response_required=true only if a concrete action, decision, or reply is needed.",
    "if actionable AND critical": "Set response_required=true only if action is needed AND the email is urgent, high-risk, or has a clear deadline.",
}


def _get_sensitivity_guidance(sensitivity: str) -> str:
    return SENSITIVITY_MAP.get(
        sensitivity.strip().lower(),
        f"Apply standard response_required rules. Sensitivity: {sensitivity}"
    )


def classify_batch(requests: List[BatchEmailItem]) -> List[BatchEmailResult]:
    """Classify up to 10 emails in a single LLM call. One prompt, one response."""
    if not requests:
        return []
    if len(requests) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE}. Received {len(requests)} requests."
        )

    email_blocks = []
    for req in requests:
        corrections = get_corrections(req.user_id, req.subject, req.bodySnippet)
        few_shot_block = build_few_shot_block(corrections)

        tag_context_lines = []
        for t in req.tags:
            prefix = "[user-defined] " if t.user_defined else ""
            desc = t.description.strip() if t.description and t.description.strip() else "No description provided"
            tag_context_lines.append(f"- {prefix}{t.name}: {desc}")
        tag_context = "\n".join(tag_context_lines)

        sensitivity_guidance = _get_sensitivity_guidance(req.sensitivity)

        block = f"""<email id="{req.id}">
Subject: {sanitize_prompt_text(req.subject)}
From: {sanitize_prompt_text(req.from_)}
Body: {sanitize_prompt_text(req.bodySnippet)}
{few_shot_block}
<categories>
{tag_context}
</categories>
<sensitivity_rule>
{sensitivity_guidance}
</sensitivity_rule>
</email>"""
        email_blocks.append(block)

    all_emails = "\n\n".join(email_blocks)

    user_prompt = f"""<batch>
{all_emails}
</batch>

Classify each email. Return only valid JSON."""

    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "batch_email_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "category": {"type": "string"},
                                "response_required": {"type": "boolean"},
                                "ai_summary": {"type": "string"},
                                "ai_action": {"type": "string"}
                            },
                            "required": ["id", "category", "response_required", "ai_summary", "ai_action"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["results"],
                "additionalProperties": False
            }
        }
    }

    try:
        completion = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        response_format=schema,
        reasoning_effort="medium",
        max_completion_tokens=20000,
        seed=42,
    )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    content = completion.choices[0].message.content
    if not content:
        finish_reason = completion.choices[0].finish_reason
        reasoning_tokens = getattr(completion.usage.completion_tokens_details, "reasoning_tokens", None) if completion.usage else None
        raise HTTPException(
            status_code=500,
            detail=f"No response from OpenAI (finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens}, max_completion_tokens=20000)"
        )

    try:
        parsed_json = json.loads(content)
        raw_results = parsed_json.get("results", [])
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from OpenAI")

    req_map = {r.id: r for r in requests}

    final_results: List[BatchEmailResult] = []
    for raw in raw_results:
        result_id = raw.get("id", "")
        req = req_map.get(result_id)

        if req is None:
            final_results.append(BatchEmailResult(
                id=result_id,
                category=raw.get("category", ""),
                response_required=raw.get("response_required", False),
                ai_summary=raw.get("ai_summary", ""),
                ai_action=raw.get("ai_action", "")
            ))
            continue

        parsed_category = raw.get("category", "")
        category = match_category(parsed_category, req.tags)
        is_actionable = is_actionable_category(category)

        ai_summary = raw.get("ai_summary", "") if is_actionable else ""
        ai_action = raw.get("ai_action", "") if is_actionable else ""

        if _is_digest_sender(req.from_):
            ai_summary = ""
            ai_action = ""

        final_results.append(BatchEmailResult(
            id=result_id,
            category=category,
            response_required=raw.get("response_required", False),
            ai_summary=ai_summary,
            ai_action=ai_action
        ))

    returned_ids = {r.id for r in final_results}
    for req in requests:
        if req.id not in returned_ids:
            final_results.append(BatchEmailResult(
                id=req.id,
                category="",
                response_required=False,
                ai_summary="",
                ai_action=""
            ))

    return final_results


@app.post("/classify", response_model=EmailClassificationResult)
def classify_email_endpoint(request: EmailRequest):
    return classify_email(request)


@app.post("/classify-batch", response_model=BatchClassifyResponse)
def classify_batch_endpoint(request: BatchClassifyRequest):
    results = classify_batch(request.requests)
    return {"results": results}

@app.post("/correct")
def store_user_correction(request: CorrectionRequest):
    save_correction(
        user_id=request.user_id,
        subject=request.subject,
        body=request.body,
        correct_label=request.correct_label,
        wrong_label=request.wrong_label
    )
    return {"status": "success", "message": "Correction saved"}

@app.post("/delete-user")
def delete_user_data(request: DeleteUserRequest):
    index.delete(filter={"user_id": {"$eq": request.user_id}})
    return {"status": "success", "message": f"All data deleted for user {request.user_id}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app")
