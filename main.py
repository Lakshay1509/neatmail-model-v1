import os
import re
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import uuid

app = FastAPI()
load_dotenv()

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

class EmailClassificationResult(BaseModel):
    category: str
    response_required: bool


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

def build_few_shot_block(corrections: list[dict]) -> str:
    if not corrections:
        return ""

    lines = ["<user_corrections>"]
    lines.append("The user has previously corrected these similar emails. Match these exactly if the current email is similar:")
    for i, c in enumerate(corrections, 1):
        wrong = f" (was wrongly labelled: {c['wrong_label']}" + ")" if c["wrong_label"] else ""
        lines.append(f"  [{i}] Email snippet: \"{c['snippet'][:100]}...\"")
        lines.append(f"       Correct label: {c['correct_label']}{wrong}")
    lines.append("</user_corrections>")

    return "\n".join(lines)


def classify_email(email_data: EmailRequest) -> EmailClassificationResult:
    corrections = get_corrections(email_data.user_id, email_data.subject, email_data.bodySnippet)
    few_shot_block = build_few_shot_block(corrections)

    tags = email_data.tags
    tag_context = "\n".join([
        f"- {t.name}: {t.description.strip() if t.description and t.description.strip() else 'No description provided'}"
        for t in tags
    ])

    # Lean system prompt: role + output contract only.
    # All dynamic context lives in the user prompt to avoid cross-message rule conflicts.
    system_prompt = """You are an email classifier. Analyze the email provided and return ONLY a JSON object.

Output format (no extra text, no markdown):
{"category": "<exact category name or empty string>", "response_required": <true|false>}

Rules:
- category: pick exactly one name from the provided list, or return "" if confidence is below 85%.
- response_required: true ONLY when the sender is a real human (not automated/no-reply), the email is directly addressed to the recipient, AND a reply/decision/action is explicitly expected. Default is false."""

    # User prompt is self-contained: email + all context needed to classify it.
    sensitivity_guidance = {
        "always draft": "Treat response_required as true for nearly all human-sent emails; false only for obvious automated/no-reply messages.",
        "if known sender AND directly addressed": "Set response_required=true only if the sender seems personally known and the email directly asks this user to respond.",
        "if actionable": "Set response_required=true only if a concrete action, decision, or reply is needed.",
        "if actionable AND critical": "Set response_required=true only if action is needed AND the email is urgent, high-risk, or has a clear deadline.",
    }.get(email_data.sensitivity.strip().lower(), f"Apply standard response_required rules. Sensitivity: {email_data.sensitivity}")

    user_prompt = f"""<email>
Subject: {email_data.subject}
From: {email_data.from_}
Body: {email_data.bodySnippet}
</email>

<categories>
{tag_context}
</categories>

<sensitivity_rule>
{sensitivity_guidance}
</sensitivity_rule>

{few_shot_block}

Classify the email. Return only valid JSON."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=40,
            seed=42,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    content = completion.choices[0].message.content
    if not content:
        raise HTTPException(status_code=500, detail="No response from OpenAI")

    try:
        parsed_json = json.loads(content)
        parsed_category = parsed_json.get("category", "")
        if not isinstance(parsed_category, str):
            parsed_category = ""
            
        def normalize(s: str) -> str:
            return re.sub(r'[^a-z0-9]', '', s.lower())
            
        normalized_parsed = normalize(parsed_category)
        
        matched_tag = None
        # 1. Try exact normalized match
        for t in tags:
            if normalize(t.name) == normalized_parsed:
                matched_tag = t
                break
                
        # 2. Fallback: try substring matching
        if not matched_tag and len(normalized_parsed) > 2:
            for t in tags:
                normalized_target = normalize(t.name)
                if normalized_target in normalized_parsed or normalized_parsed in normalized_target:
                    matched_tag = t
                    break

        category = matched_tag.name if matched_tag else ""
        
        response_required = parsed_json.get("response_required", False)
        if not isinstance(response_required, bool):
            response_required = False

        return EmailClassificationResult(
            category=category,
            response_required=response_required
        )
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from OpenAI")

@app.post("/classify", response_model=EmailClassificationResult)
def classify_email_endpoint(request: EmailRequest):
    return classify_email(request)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app")
