import os
import io
import json
import re
import httpx
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
    function_tool,
    ModelSettings,
)

# ─────────────────────────────────────────
# ENV & CLIENT SETUP
# ─────────────────────────────────────────

load_dotenv()

API_KEY     = os.getenv("GEMINI_API_KEY", "")
MODEL       = os.getenv("MODEL", "gemini-2.5-flash")
OCR_API_KEY = os.getenv("OCR_API_KEY", "K88317972588957")   # ocr.space API key

client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=API_KEY,
)

set_default_openai_api("chat_completions")
set_default_openai_client(client)
set_tracing_disabled(True)

# ─────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────

app = FastAPI(
    title="🎯 AI Document Structured Data Extraction Engine",
    description=(
        "Intelligent Document Processing (IDP) — convert unstructured PDFs, images, "
        "emails, and scanned documents into clean, structured JSON automatically."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────

class ExtractionResponse(BaseModel):
    document_type: str
    confidence: str
    structured_data: Dict[str, Any]
    raw_text_preview: str
    processing_notes: List[str]

class TextInput(BaseModel):
    text: str
    document_type: Optional[str] = None

class BatchTextInput(BaseModel):
    documents: List[TextInput]

# ─────────────────────────────────────────
# OCR.SPACE API — TEXT EXTRACTION
# ─────────────────────────────────────────

OCR_SPACE_URL = "https://api.ocr.space/parse/image"

# Supported file extensions routed through OCR.space
OCR_SUPPORTED_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".bmp", ".webp")

MIME_MAP = {
    "pdf":  "application/pdf",
    "png":  "image/png",
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "gif":  "image/gif",
    "tiff": "image/tiff",
    "tif":  "image/tiff",
    "bmp":  "image/bmp",
    "webp": "image/webp",
}


async def ocr_space_extract(file_bytes: bytes, filename: str) -> str:
    """
    Send a file to OCR.space API via multipart upload and return the extracted text.

    - OCREngine 2: handles complex layouts, tables, scanned documents.
    - isTable=true: improves invoice/receipt row detection.
    - detectOrientation: auto-corrects rotated scans.
    - No Tesseract dependency required.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "png"
    mime_type = MIME_MAP.get(ext, "application/octet-stream")

    async with httpx.AsyncClient(timeout=90) as http:
        response = await http.post(
            OCR_SPACE_URL,
            data={
                "apikey":            OCR_API_KEY,
                "language":          "eng",
                "isOverlayRequired": "false",
                "detectOrientation": "true",
                "scale":             "true",
                "isTable":           "true",   # better for invoices, receipts, forms
                "OCREngine":         "2",       # engine 2 = superior accuracy
            },
            files={"file": (filename, file_bytes, mime_type)},
        )

    if response.status_code != 200:
        raise HTTPException(
            502,
            f"OCR.space API returned HTTP {response.status_code}: {response.text[:300]}"
        )

    result = response.json()

    if result.get("IsErroredOnProcessing"):
        error_msg = result.get("ErrorMessage", ["Unknown OCR error"])
        if isinstance(error_msg, list):
            error_msg = " | ".join(str(m) for m in error_msg)
        raise HTTPException(422, f"OCR.space processing error: {error_msg}")

    parsed_results = result.get("ParsedResults") or []
    if not parsed_results:
        raise HTTPException(422, "OCR.space returned empty results — document may be blank or unreadable")

    # Join text from all pages/results
    full_text = "\n\n".join(
        page.get("ParsedText", "").strip()
        for page in parsed_results
        if page.get("ParsedText", "").strip()
    )

    return full_text.strip()


async def extract_text_from_bytes(filename: str, content: bytes) -> str:
    """
    Route to the correct extraction strategy based on file extension:
      - PDF / images  → OCR.space API (cloud OCR, no local dependencies)
      - Plain text    → decode directly (UTF-8)
    """
    fname = filename.lower()

    # Plain text formats — read directly, no OCR needed
    if fname.endswith((".txt", ".eml", ".csv", ".md")):
        return content.decode("utf-8", errors="ignore").strip()

    # Documents and images — route through OCR.space
    if fname.endswith(OCR_SUPPORTED_EXTS):
        return await ocr_space_extract(content, filename)

    raise HTTPException(
        400,
        f"Unsupported file type: '{filename}'. "
        f"Accepted: PDF, PNG, JPG, JPEG, GIF, TIFF, BMP, WEBP, TXT, EML, CSV, MD"
    )



# ─────────────────────────────────────────
# DOCUMENT SCHEMAS (for extraction hints)
# ─────────────────────────────────────────

DOCUMENT_SCHEMAS = {
    "invoice": {
        "fields": ["vendor", "invoice_number", "date", "due_date", "currency",
                   "subtotal", "tax", "total_amount", "bill_to", "ship_to", "line_items[]"],
        "line_item_fields": ["item", "description", "qty", "unit_price", "total"]
    },
    "receipt": {
        "fields": ["store_name", "date", "time", "items[]", "subtotal", "tax",
                   "total", "payment_method", "transaction_id"]
    },
    "purchase_order": {
        "fields": ["po_number", "buyer", "vendor", "date", "delivery_date",
                   "items[]", "total_amount", "currency", "shipping_address", "terms"]
    },
    "resume": {
        "fields": ["name", "email", "phone", "location", "linkedin", "summary",
                   "skills[]", "education[]", "experience[]", "certifications[]", "languages[]"]
    },
    "job_application": {
        "fields": ["applicant_name", "position_applied", "date", "email", "phone",
                   "cover_letter_summary", "references[]"]
    },
    "contract": {
        "fields": ["contract_title", "parties[]", "effective_date", "termination_date",
                   "jurisdiction", "key_obligations[]", "payment_terms", "penalties", "signatures[]"]
    },
    "agreement": {
        "fields": ["agreement_type", "parties[]", "date", "duration",
                   "terms[]", "governing_law", "signatures[]"]
    },
    "email": {
        "fields": ["from", "to", "cc", "subject", "date", "body_summary",
                   "action_items[]", "sentiment", "priority"]
    },
    "meeting_notes": {
        "fields": ["meeting_title", "date", "attendees[]", "agenda[]",
                   "decisions[]", "action_items[]", "next_meeting"]
    },
    "report": {
        "fields": ["title", "author", "date", "executive_summary", "sections[]",
                   "key_findings[]", "recommendations[]", "conclusion"]
    },
    "form": {
        "fields": ["form_title", "form_id", "submitted_by", "date",
                   "fields_data", "signatures[]"]
    },
}

SCHEMA_HINT_TEXT = "\n".join(
    f"- **{k}**: fields → {', '.join(v['fields'])}"
    for k, v in DOCUMENT_SCHEMAS.items()
)

# ─────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────

classification_agent = Agent(
    name="Document Classifier",
    instructions=f"""
You are an expert document classifier for an Intelligent Document Processing system.

Analyze the provided document text and classify it into exactly ONE of these types:
invoice | receipt | purchase_order | resume | job_application | contract | agreement | email | meeting_notes | report | form | unknown

Rules:
- Return ONLY the classification label — nothing else
- Choose the most specific match (e.g. 'invoice' over 'form')
- If truly ambiguous, return 'unknown'

Known document schemas:
{SCHEMA_HINT_TEXT}
""",
    model=MODEL,
)

extraction_agent = Agent(
    name="Structured Data Extractor",
    instructions="""
You are a world-class document data extraction AI for an Intelligent Document Processing (IDP) platform.

Your job: given a document type and raw document text, extract ALL relevant structured data and return it as valid JSON.

Strict rules:
1. Return ONLY valid JSON — no markdown fences, no prose, no explanation
2. Use the document type to determine which fields to extract
3. Normalize all dates to ISO 8601 format (YYYY-MM-DD)
4. All monetary values as numbers (not strings)
5. Arrays for repeating items (line_items, skills, experience, etc.)
6. Missing fields → null (never omit them)
7. Infer currency from symbols ($=USD, €=EUR, £=GBP, etc.)
8. For resumes: parse experience as [{company, title, start_date, end_date, description}]
9. For invoices: parse line_items as [{item, description, qty, unit_price, total}]
10. For contracts: extract key_obligations as a list of strings
11. Be thorough — extract every piece of data visible in the text

Field naming: use snake_case for all keys.
""",
    model=MODEL,
    model_settings=ModelSettings(temperature=0.1),
)

validation_agent = Agent(
    name="Data Validator",
    instructions="""
You are a data quality validator for extracted document data.

Given a document type and extracted JSON data, check for:
1. Missing critical fields
2. Invalid date formats
3. Math errors (do line items sum to total?)
4. Suspicious or obviously wrong values
5. Completeness of arrays

Return a JSON object with:
{
  "confidence": "high" | "medium" | "low",
  "notes": ["list of issues or confirmations"]
}

Return ONLY valid JSON — no markdown, no prose.
""",
    model=MODEL,
)

# ─────────────────────────────────────────
# CORE PROCESSING PIPELINE
# ─────────────────────────────────────────

def clean_json_output(raw: str) -> str:
    """Strip markdown fences and extract JSON from model output."""
    raw = raw.strip()
    # Remove ```json ... ``` fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()

async def process_document(text: str, forced_type: Optional[str] = None) -> tuple:
    notes = []

    # Step 1: Classification
    if forced_type and forced_type.strip():
        doc_type = forced_type.strip().lower()
        notes.append(f"Document type manually set to: {doc_type}")
    else:
        clf_result = await Runner.run(classification_agent, text[:4000])
        doc_type = clf_result.final_output.strip().lower()
        notes.append(f"Auto-detected document type: {doc_type}")

    # Step 2: Schema hint
    schema = DOCUMENT_SCHEMAS.get(doc_type, {})
    schema_hint = ""
    if schema:
        schema_hint = f"\nExpected fields for {doc_type}: {', '.join(schema.get('fields', []))}"

    # Step 3: Extraction
    extraction_prompt = f"""Document Type: {doc_type}
{schema_hint}

--- DOCUMENT TEXT START ---
{text}
--- DOCUMENT TEXT END ---
"""
    ext_result = await Runner.run(extraction_agent, extraction_prompt)
    raw_json = clean_json_output(ext_result.final_output)

    try:
        structured = json.loads(raw_json)
    except json.JSONDecodeError as e:
        notes.append(f"JSON parse warning: {e} — attempting recovery")
        # Try to find JSON object in output
        match = re.search(r"\{.*\}", raw_json, re.DOTALL)
        if match:
            try:
                structured = json.loads(match.group())
            except:
                raise HTTPException(500, f"Model returned invalid JSON: {raw_json[:300]}")
        else:
            raise HTTPException(500, "Could not extract valid JSON from model output")

    # Step 4: Validation
    val_prompt = f"""Document Type: {doc_type}
Extracted Data: {json.dumps(structured, indent=2)}"""
    val_result = await Runner.run(validation_agent, val_prompt)
    val_raw = clean_json_output(val_result.final_output)

    confidence = "medium"
    try:
        val_data = json.loads(val_raw)
        confidence = val_data.get("confidence", "medium")
        val_notes = val_data.get("notes", [])
        notes.extend(val_notes)
    except:
        notes.append("Validation step returned non-JSON; skipping detailed validation")

    return doc_type, structured, confidence, notes

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "AI Document Structured Data Extraction Engine",
        "version": "2.0.0",
        "status": "running",
        "supported_types": list(DOCUMENT_SCHEMAS.keys()),
        "endpoints": {
            "POST /extract": "Upload file (PDF, image, txt, eml)",
            "POST /extract/text": "Submit raw text",
            "POST /extract/batch": "Batch process multiple text documents",
            "GET /schemas": "List all supported document schemas",
            "GET /health": "Health check",
        }
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model": MODEL}

@app.get("/schemas", tags=["Schemas"])
def get_schemas():
    return {
        "supported_document_types": DOCUMENT_SCHEMAS,
        "total": len(DOCUMENT_SCHEMAS),
    }

# ─────────────────────────────────────────
# FILE UPLOAD ENDPOINT
# ─────────────────────────────────────────

@app.post("/extract", response_model=ExtractionResponse, tags=["Extraction"])
async def extract_from_file(
    file: UploadFile = File(..., description="PDF, PNG, JPG, TXT, or EML file"),
    document_type: Optional[str] = Form(
        None,
        description="Force document type (invoice, resume, contract, etc.) or leave blank for auto-detect"
    ),
):
    """
    Upload a document file and extract structured data automatically.
    
    Supports: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP, TXT, EML, CSV, MD
    """
    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty")

    text = await extract_text_from_bytes(file.filename or "file.txt", content)

    if not text.strip():
        raise HTTPException(400, "No readable text found in document. For scanned PDFs/images, ensure OCR dependencies are installed.")

    doc_type, structured, confidence, notes = await process_document(text, document_type)

    return ExtractionResponse(
        document_type=doc_type,
        confidence=confidence,
        structured_data=structured,
        raw_text_preview=text[:500] + ("..." if len(text) > 500 else ""),
        processing_notes=notes,
    )

# ─────────────────────────────────────────
# RAW TEXT ENDPOINT
# ─────────────────────────────────────────

@app.post("/extract/text", response_model=ExtractionResponse, tags=["Extraction"])
async def extract_from_text(data: TextInput):
    """
    Submit raw text directly (email body, copied document content, etc.)
    """
    if not data.text.strip():
        raise HTTPException(400, "Text input is empty")

    doc_type, structured, confidence, notes = await process_document(
        data.text, data.document_type
    )

    return ExtractionResponse(
        document_type=doc_type,
        confidence=confidence,
        structured_data=structured,
        raw_text_preview=data.text[:500] + ("..." if len(data.text) > 500 else ""),
        processing_notes=notes,
    )

# ─────────────────────────────────────────
# BATCH ENDPOINT
# ─────────────────────────────────────────

@app.post("/extract/batch", tags=["Extraction"])
async def extract_batch(data: BatchTextInput):
    """
    Process multiple text documents in one request.
    Returns a list of extraction results.
    """
    if not data.documents:
        raise HTTPException(400, "No documents provided")
    if len(data.documents) > 10:
        raise HTTPException(400, "Maximum 10 documents per batch")

    results = []
    for i, doc in enumerate(data.documents):
        try:
            doc_type, structured, confidence, notes = await process_document(
                doc.text, doc.document_type
            )
            results.append({
                "index": i,
                "status": "success",
                "document_type": doc_type,
                "confidence": confidence,
                "structured_data": structured,
                "processing_notes": notes,
            })
        except Exception as e:
            results.append({
                "index": i,
                "status": "error",
                "error": str(e),
            })

    return {"total": len(data.documents), "results": results}

# ─────────────────────────────────────────
# EXCEPTION HANDLERS
# ─────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )