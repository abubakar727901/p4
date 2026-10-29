import base64
import json
import os
from io import BytesIO
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError
from PIL import Image, ImageOps
from pypdf import PdfReader
from pypdf.errors import PdfReadError


MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 25
MAX_IMAGE_DIMENSION = 2000
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
LANGUAGES = {
    "English": "English",
    "Urdu": "Urdu",
    "Roman Urdu": "Roman Urdu (Urdu written using Latin script)",
}

st.set_page_config(
    page_title="Prescription Explainer AI",
    page_icon="✚",
    layout="centered",
)

load_dotenv()

st.markdown(
    """
    <style>
    :root {
        --emerald: #087f5b;
        --emerald-light: #e8f5ef;
        --slate: #253342;
        --muted: #627184;
        --border: #dce5e2;
    }
    .stApp { background: #ffffff; color: var(--slate); }
    .main .block-container { max-width: 900px; padding: 2.5rem 1.5rem 4rem; }
    h1, h2, h3 { color: var(--slate); letter-spacing: -0.025em; }
    .hero-kicker {
        color: var(--emerald); font-weight: 700; font-size: .78rem;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .35rem;
    }
    .hero-copy { color: var(--muted); font-size: 1.05rem; line-height: 1.65; }
    [data-testid="stFileUploader"] {
        border: 1px dashed #9bbcaf; border-radius: 14px; padding: .6rem;
        background: #fbfdfc;
    }
    div.stButton > button[kind="primary"] {
        background: var(--emerald); border-color: var(--emerald);
        border-radius: 9px; min-height: 2.9rem; font-weight: 650;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #066b4d; border-color: #066b4d;
    }
    .disclaimer {
        border-left: 4px solid var(--emerald); background: var(--emerald-light);
        border-radius: 8px; padding: .9rem 1rem; color: #234438;
        line-height: 1.55; margin: 1rem 0;
    }
    .section-label {
        color: var(--emerald); font-weight: 700; font-size: .78rem;
        letter-spacing: .09em; text-transform: uppercase; margin: .3rem 0 .7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _compact_image(image: Image.Image) -> bytes:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
    output = BytesIO()
    image.save(output, format="JPEG", quality=88, optimize=True)
    return output.getvalue()


def _image_content(data: bytes, filename: str) -> list[dict[str, str]]:
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            expected_extensions = {
                "PNG": {"png"},
                "JPEG": {"jpg", "jpeg"},
                "WEBP": {"webp"},
            }
            suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if suffix not in expected_extensions.get(image.format, set()):
                raise ValueError("The file extension does not match its image format.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("The image resolution is too large to process.")
            prepared = _compact_image(image)
    except (Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ValueError(
            "The image could not be opened, is too large, or its file type does not match. "
            "Please try a valid PNG, JPG, JPEG, or WEBP file."
        ) from exc

    encoded = base64.b64encode(prepared).decode("ascii")
    return [
        {
            "type": "text",
            "text": f"Uploaded prescription image: {filename}",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}",
                "detail": "high",
            },
        },
    ]


def _pdf_content(data: bytes, filename: str) -> list[dict[str, str]]:
    try:
        reader = PdfReader(BytesIO(data), strict=False)
    except (PdfReadError, OSError, ValueError) as exc:
        raise ValueError("This PDF could not be opened. Please upload a valid, unencrypted PDF.") from exc

    if reader.is_encrypted:
        raise ValueError("This PDF is password-protected. Please upload an unencrypted PDF.")
    if len(reader.pages) == 0:
        raise ValueError("The PDF contains no pages.")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"This PDF has more than {MAX_PDF_PAGES} pages. Please upload a shorter document.")

    page_text: list[str] = []
    try:
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                page_text.append(f"\n--- PDF page {page_number} text ---\n{text}")
    except (PdfReadError, OSError, ValueError) as exc:
        raise ValueError("Selectable text could not be extracted from this PDF.") from exc

    if page_text:
        description = (
            "The attached PDF may contain selectable text and scanned or handwritten content. "
            "Compare its pages with this selectable text, and use the PDF page images to check "
            "handwriting or unclear printed text. Use only legible content; never guess.\n"
            "Selectable text by page:\n"
            + "".join(page_text)
        )
    else:
        description = (
            "No selectable text was found in the attached PDF. Inspect its page images only where "
            "the writing is legible; mark uncertain text as unclear and never guess."
        )

    encoded = base64.b64encode(data).decode("ascii")
    return [
        {"type": "text", "text": description},
        {
            "type": "file",
            "file": {
                "filename": filename,
                "file_data": f"data:application/pdf;base64,{encoded}",
            },
        },
    ]


def _build_prompt(language: str) -> str:
    return f"""
Explain this prescription in {language}. This is an educational transcription and explanation task,
not diagnosis or prescribing. Inspect only the supplied document. Do not infer a medicine's purpose
or diagnosis unless the document explicitly says it. Never guess medicine names, strengths, doses,
frequencies, duration, food instructions, or timings. Preserve each readable instruction exactly as
written in its corresponding *_as_written field. Use an empty string for absent or unreadable fields,
and explain uncertainty in unclear_fields. Separate what is clearly readable from what is uncertain.
In plain_language_explanation, explain only the written instruction in simple {language}; do not
recommend starting, stopping, changing, doubling, or substituting medicines. Do not calculate a dose.
Mention urgent concerns only when clearly supported by the document; direct the reader to a qualified
healthcare professional or emergency service when warranted. Do not falsely reassure.

Return ONLY a valid JSON object with exactly these top-level keys:
{{
  "medicines": [
    {{
      "name_as_written": "string",
      "strength_as_written": "string",
      "dosage_as_written": "string",
      "frequency_as_written": "string",
      "duration_as_written": "string",
      "timing_food_as_written": "string",
      "plain_language_explanation": "string",
      "unclear_fields": ["string"]
    }}
  ],
  "written_instructions": ["string"],
  "unclear_information": ["string"],
  "safety_notes": ["string"],
  "summary": "string"
}}

List each medicine only when its name can be identified with reasonable confidence. If no medicine can
be confidently identified, return an empty medicines list and describe what specifically is unclear.
Do not include Markdown fences or additional keys. The summary must be a faithful, concise restatement
of these results in {language}, with no new medical advice.
""".strip()


def _validate_result(result: Any) -> dict[str, Any]:
    expected = {
        "medicines",
        "written_instructions",
        "unclear_information",
        "safety_notes",
        "summary",
    }
    if not isinstance(result, dict) or set(result) != expected:
        raise ValueError("The AI returned an unexpected response format. Please try again.")
    if not isinstance(result["medicines"], list) or not all(
        isinstance(item, dict) for item in result["medicines"]
    ):
        raise ValueError("The AI returned invalid medicine details. Please try again.")
    for medicine in result["medicines"]:
        required = {
            "name_as_written",
            "strength_as_written",
            "dosage_as_written",
            "frequency_as_written",
            "duration_as_written",
            "timing_food_as_written",
            "plain_language_explanation",
            "unclear_fields",
        }
        if set(medicine) != required or not all(
            isinstance(medicine[key], str)
            for key in required - {"unclear_fields"}
        ) or not isinstance(medicine["unclear_fields"], list) or not all(
            isinstance(value, str) for value in medicine["unclear_fields"]
        ):
            raise ValueError("The AI returned incomplete or invalid medicine details. Please try again.")
        if not medicine["name_as_written"].strip():
            raise ValueError("The AI returned a medicine without a readable name. Please try again.")
    for key in ("written_instructions", "unclear_information", "safety_notes"):
        if not isinstance(result[key], list) or not all(isinstance(item, str) for item in result[key]):
            raise ValueError("The AI returned an invalid response format. Please try again.")
    if not isinstance(result["summary"], str) or not result["summary"].strip():
        raise ValueError("The AI did not return a usable summary. Please try again.")
    return result


def _explain(data: bytes, filename: str, language: str) -> dict[str, Any]:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix == "pdf":
        document_content = _pdf_content(data, filename)
    elif suffix in SUPPORTED_IMAGE_TYPES:
        document_content = _image_content(data, filename)
    else:
        raise ValueError("Unsupported file type. Upload a PDF, PNG, JPG, JPEG, or WEBP file.")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Add it to your local .env file and restart the app.")
    model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
    if not model:
        raise ValueError("OPENROUTER_MODEL is empty. Set it to a vision-capable model available on OpenRouter.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": _build_prompt(language)}]
                + document_content,
            }
        ],
        response_format={"type": "json_object"},
    )
    response_text = response.choices[0].message.content
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("The AI returned no explanation. Please try again.")
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("The AI response could not be read safely. Please try again.") from exc
    return _validate_result(parsed)


def _display_values(title: str, values: list[str], empty_message: str) -> None:
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)
    if values:
        for value in values:
            st.text(f"• {value}")
    else:
        st.caption(empty_message)


def _copy_summary(summary: str) -> None:
    encoded_summary = base64.b64encode(summary.encode("utf-8")).decode("ascii")
    st.components.v1.html(
        f"""
        <button id="copy-summary" style="
          background:#087f5b;color:white;border:0;border-radius:8px;padding:10px 16px;
          font:600 14px system-ui;cursor:pointer">
          Copy Summary
        </button>
        <span id="copy-status" role="status" style="
          color:#253342;font:13px system-ui;margin-left:10px"></span>
        <script>
          const encodedSummary = "{encoded_summary}";
          const binarySummary = atob(encodedSummary);
          const summaryBytes = Uint8Array.from(binarySummary, char => char.charCodeAt(0));
          const summary = new TextDecoder().decode(summaryBytes);
          const button = document.getElementById("copy-summary");
          const status = document.getElementById("copy-status");
          button.addEventListener("click", async () => {{
            try {{
              await navigator.clipboard.writeText(summary);
              status.textContent = "Summary copied.";
            }} catch (error) {{
              const textArea = document.createElement("textarea");
              textArea.value = summary;
              textArea.style.position = "fixed";
              textArea.style.opacity = "0";
              document.body.appendChild(textArea);
              textArea.select();
              const copied = document.execCommand("copy");
              textArea.remove();
              status.textContent = copied
                ? "Summary copied."
                : "Copy was blocked by the browser. Select and copy the summary below.";
            }}
          }});
        </script>
        """,
        height=48,
    )
    st.text_area("Generated summary", value=summary, height=110, disabled=True, label_visibility="collapsed")


st.markdown('<div class="hero-kicker">Understand your prescription</div>', unsafe_allow_html=True)
st.title("Prescription Explainer AI")
st.markdown(
    '<p class="hero-copy">Upload a prescription to see a plain-language explanation of what is '
    "readable. Unclear details are called out instead of guessed.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer"><strong>Educational use only.</strong> This tool does not diagnose or '
    "prescribe. Have every interpretation confirmed by a qualified pharmacist or doctor. "
    "Never change how you take a medicine based on this explanation.</div>",
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("Upload prescription")
    st.caption(
        "The document is sent to OpenRouter for analysis. This app does not save uploads to a database."
    )
    uploaded_file = st.file_uploader(
        "Choose a prescription image or PDF",
        type=SUPPORTED_IMAGE_TYPES + ["pdf"],
        max_upload_size=15,
        help="PDF, PNG, JPG, JPEG, or WEBP · Maximum 15 MB · PDFs up to 25 pages",
    )
    language = st.selectbox("Explanation language", options=list(LANGUAGES))
    explain_clicked = st.button(
        "Explain Prescription",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

if explain_clicked and uploaded_file is not None:
    st.session_state.pop("prescription_explanation", None)
    st.session_state.pop("prescription_file_id", None)
    raw_file = uploaded_file.getvalue()
    if len(raw_file) > MAX_FILE_BYTES:
        st.error("This file is larger than 15 MB. Please upload a smaller file.")
    else:
        try:
            with st.spinner("Reading the document and preparing an explanation…"):
                st.write("Checking the uploaded file.")
                result = _explain(raw_file, uploaded_file.name, LANGUAGES[language])
            st.session_state["prescription_explanation"] = result
            st.session_state["prescription_language"] = language
            st.session_state["prescription_file_id"] = uploaded_file.file_id
        except ValueError as exc:
            st.error(str(exc))
        except APIConnectionError:
            st.error("Could not connect to OpenRouter. Check your connection and try again.")
        except APIStatusError as exc:
            st.error(f"OpenRouter returned an error (HTTP {exc.status_code}). Please try again later.")
        except OpenAIError:
            st.error("The OpenRouter request failed. Check your API key, model ID, and try again.")

result = st.session_state.get("prescription_explanation")
if (
    result
    and uploaded_file is not None
    and st.session_state.get("prescription_file_id") == uploaded_file.file_id
    and st.session_state.get("prescription_language") == language
):
    st.divider()
    st.header("Prescription explanation")
    if result["medicines"]:
        st.markdown('<div class="section-label">Medicine details</div>', unsafe_allow_html=True)
        for index, medicine in enumerate(result["medicines"], start=1):
            heading = medicine["name_as_written"] or f"Medicine {index} (name unclear)"
            with st.container(border=True):
                st.subheader(heading)
                fields = [
                    ("Strength", medicine["strength_as_written"]),
                    ("Dosage", medicine["dosage_as_written"]),
                    ("Frequency", medicine["frequency_as_written"]),
                    ("Duration", medicine["duration_as_written"]),
                    ("Food or timing", medicine["timing_food_as_written"]),
                ]
                for label, value in fields:
                    if value:
                        st.markdown(f"**{label} (as written):**")
                        st.text(value)
                    else:
                        st.markdown(f"**{label}:** Not stated or not clearly readable")
                if medicine["plain_language_explanation"]:
                    st.markdown("**In simple language**")
                    st.write(medicine["plain_language_explanation"])
                _display_values(
                    "Unclear details for this medicine",
                    medicine["unclear_fields"],
                    "No additional unclear fields were identified.",
                )
    else:
        st.info("No medicine name could be identified confidently from this document.")

    _display_values(
        "Written instructions",
        result["written_instructions"],
        "No separate written instructions were identified.",
    )
    _display_values(
        "Unclear or missing information",
        result["unclear_information"],
        "No additional unclear information was reported.",
    )
    _display_values(
        "Safety notes",
        result["safety_notes"],
        "No specific safety note was identified in the document. This does not confirm that the prescription is appropriate.",
    )
    st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
    st.text(result["summary"])
    _copy_summary(result["summary"])
    st.markdown(
        '<div class="disclaimer">Please confirm the reading and all medicine instructions with a '
        "qualified pharmacist or doctor before acting on them.</div>",
        unsafe_allow_html=True,
    )
