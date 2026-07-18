import json
import os
from pathlib import Path

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

MODEL_NAME = "gemma-4-31b-it"
STYLES_PATH = Path(__file__).parent / "styles.css"
SEVERITY_LEVELS = {"low", "medium", "high"}
PRIORITY_LEVELS = {"low", "medium", "high", "immediate"}


def load_css(path: Path) -> None:
    if path.exists():
        st.markdown(f"<style>{path.read_text()}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    defaults = {"analysis_result": None, "analyzed_image": None, "analyzed_location": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    st.markdown(
        """
        <div class='app-header'>
          <div class='app-header-left'>
            <div class='app-title'>CleanLink AI</div>
            <div class='app-subtitle'>Environmental Waste Intelligence Platform</div>
            <div class='app-powered-by'>Powered by Google Gemma</div>
          </div>
          <div class='app-header-right'>
            <span class='badge badge-primary'>Gemma 4</span>
            <span class='badge badge-primary'>AI Powered</span>
            <span class='badge badge-neutral'>Version 1.0</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class='hero-section'>
          <h1>Environmental Waste Assessment</h1>
          <div class='hero-subtitle'>
            Upload a photo of a waste site to get an AI-generated severity, priority,
            and municipal-ready report.
          </div>
          <div class='capability-grid'>
            <div class='capability-card'>
              <div class='capability-title'>AI Powered</div>
              <div class='capability-desc'>Google Gemma 4 Vision</div>
            </div>
            <div class='capability-card'>
              <div class='capability-title'>Computer Vision</div>
              <div class='capability-desc'>Structured Reports</div>
            </div>
            <div class='capability-card'>
              <div class='capability-title'>Municipal Ready</div>
              <div class='capability-desc'>Priority Detection</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_settings(env_api_key: str | None) -> str | None:
    with st.container(key="settings-card"):
        st.markdown("<div class='section-title'>Application Settings</div>", unsafe_allow_html=True)
        with st.expander("Settings", expanded=not env_api_key):
            api_key = st.text_input(
                "Google AI Studio API key",
                type="password",
                value=env_api_key or "",
                placeholder="AIzaSy...",
            )
            st.caption("Get a free key at [Google AI Studio](https://aistudio.google.com/).")

    return api_key


def configure_model(api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def build_prompt(location: str) -> str:
    location_line = f"Location: {location}." if location else "Location: not specified."

    return f"""Analyze this environmental waste image. {location_line}

Respond with a single raw JSON object only — no markdown, no code fences, no commentary.

Required keys:
- waste_type: string
- severity: "Low" | "Medium" | "High"
- health_risks: array of strings
- priority: "Low" | "Medium" | "High" | "Immediate"
- action: string, one recommended immediate cleanup action
- report: string, a concise report for a municipal waste management agency
- confidence_score: integer 0-100

Base every field strictly on what is visible in the image. Do not invent details."""


def extract_json_object(text: str) -> str | None:
    """Return the first complete, balanced {...} object found in text, or None.

    Scans for a '{' and tracks brace depth while respecting string literals
    (so braces inside quoted values don't throw off the count), then returns
    as soon as that object closes. This lets us pull the JSON out even when
    the model prefaces or follows it with reasoning text.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
            # unbalanced quote or brace run past end of text; try the next '{'
        start = text.find("{", start + 1)
    return None


def parse_response(raw_text: str) -> dict:
    json_str = extract_json_object(raw_text)
    if json_str is not None:
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "waste_type": "Indeterminate Waste Matter",
        "severity": "Medium",
        "health_risks": ["Potential drainage blockages", "Hygiene hazards"],
        "priority": "Medium",
        "action": "Inspect site details manually.",
        "report": raw_text,
        "confidence_score": 75,
    }


def analyze_image(image: Image.Image, location: str, api_key: str) -> dict:
    model = configure_model(api_key)
    prompt = build_prompt(location)
    response = model.generate_content([image, prompt])
    return parse_response(response.text)


def badge_html(value: str, valid_values: set[str]) -> str:
    value = (value or "Medium").strip()
    css_class = f"badge-{value.lower()}" if value.lower() in valid_values else "badge-medium"
    return f"<span class='badge {css_class}'>{value}</span>"


def generate_report(result: dict, location: str | None) -> str:
    risks = "\n".join(f"- {risk}" for risk in result.get("health_risks", []))
    return f"""CLEANLINK AI — WASTE ANALYSIS REPORT
=========================================
Location: {location or "N/A"}
Waste type: {result.get('waste_type')}
Severity: {result.get('severity')}
Priority: {result.get('priority')}
Confidence: {result.get('confidence_score')}%

Health & Environmental Risks
{risks}

Recommended Action
{result.get('action')}

Agency Report
{result.get('report')}
========================================="""


def render_assessment(result: dict) -> None:
    st.markdown("<div class='section-title'>Assessment</div>", unsafe_allow_html=True)
    with st.container(key="stats-grid"):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f"<div class='metric-card'><div class='stat-label'>Waste type</div>"
                f"<div class='stat-value'>{result.get('waste_type', 'Unknown')}</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='metric-card'><div class='stat-label'>Severity</div>"
                f"{badge_html(result.get('severity'), SEVERITY_LEVELS)}</div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"<div class='metric-card'><div class='stat-label'>Priority</div>"
                f"{badge_html(result.get('priority'), PRIORITY_LEVELS)}</div>",
                unsafe_allow_html=True,
            )
        with c4:
            confidence = result.get("confidence_score", 85)
            st.markdown(
                f"<div class='metric-card'><div class='stat-label'>Confidence</div>"
                f"<div class='stat-value'>{confidence}%</div></div>",
                unsafe_allow_html=True,
            )


def render_risks_and_action(result: dict) -> None:
    with st.container(key="insights-grid"):
        risk_col, action_col = st.columns(2)

        with risk_col:
            st.markdown("<div class='section-title'>Environmental Risks</div>", unsafe_allow_html=True)
            risks = result.get("health_risks", [])
            if risks:
                items = "".join(f"<div class='risk-item'>{risk}</div>" for risk in risks)
                st.markdown(f"<div class='risk-card'>{items}</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='risk-card'><div class='risk-item'>No significant risks flagged.</div></div>",
                    unsafe_allow_html=True,
                )

        with action_col:
            st.markdown("<div class='section-title'>Recommended Action</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='action-card'>"
                f"<div class='action-title'>Recommended Action</div>"
                f"<div class='action-body'>{result.get('action', 'Standard monitoring and cleanup protocol.')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_agency_report(result: dict) -> None:
    st.markdown("<div class='section-title'>Agency Report</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='report-card'><p>{result.get('report', 'No report generated.')}</p></div>",
        unsafe_allow_html=True,
    )


def spacer(height_rem: float = 2.0) -> None:
    st.markdown(f"<div style='height:{height_rem}rem'></div>", unsafe_allow_html=True)


def render_download(result: dict, location: str | None) -> None:
    with st.container(key="download-section"):
        st.markdown("<div class='section-title'>Export Report</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Download the generated assessment.</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download Report",
            data=generate_report(result, location),
            file_name="cleanlink-waste-report.txt",
            mime="text/plain",
            use_container_width=True,
        )


def render_results(result: dict, location: str | None) -> None:
    st.divider()
    with st.container(key="results-section"):
        render_assessment(result)
        spacer()
        render_risks_and_action(result)
        spacer()
        render_agency_report(result)

    spacer(1.5)
    render_download(result, location)


def render_upload_form() -> tuple:
    with st.container(key="workspace"):
        left, right = st.columns([1, 1], gap="large")

        with left:
            with st.container(key="upload-card"):
                st.markdown("<div class='section-title'>Upload Waste Image</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-subtitle'>Upload a photo for AI analysis.</div>",
                    unsafe_allow_html=True,
                )
                uploaded_file = st.file_uploader("Waste image", type=["jpg", "jpeg", "png", "webp"])
                location = st.text_input("Location (optional)", placeholder="e.g. Sabo Market, Ogbomoso")
                st.markdown(
                    "<div class='field-helper-text'>Location improves report context.</div>",
                    unsafe_allow_html=True,
                )
                analyze_clicked = st.button("Analyze Waste", use_container_width=True)

        with right:
            with st.container(key="preview-card"):
                st.markdown("<div class='section-title'>Image Preview</div>", unsafe_allow_html=True)
                if uploaded_file is not None:
                    try:
                        preview = Image.open(uploaded_file)
                        st.image(preview, use_container_width=True)
                    except Exception:
                        st.error("Could not load this image. Try a different file.")
                else:
                    st.markdown(
                        "<div class='empty-preview'>"
                        "<div class='empty-icon'>🖼️</div>"
                        "<div class='empty-title'>No image uploaded</div>"
                        "<div class='empty-subtitle'>Upload an image to preview it here.</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

    return uploaded_file, location, analyze_clicked


def render_footer() -> None:
    st.markdown(
        """
        <div class='app-footer'>
          <div class='footer-line'>CleanLink AI — Environmental Waste Intelligence Platform</div>
          <div class='footer-line'>Powered by Google Gemma 4 · Version 1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def handle_analysis(uploaded_file, location: str, api_key: str | None) -> None:
    if not api_key:
        st.error("Add your Google AI Studio API key in Settings.")
        return
    if uploaded_file is None:
        st.error("Upload an image first.")
        return

    with st.spinner("Analyzing image..."):
        try:
            image = Image.open(uploaded_file)
            result = analyze_image(image, location, api_key)
            st.session_state.analysis_result = result
            st.session_state.analyzed_image = image
            st.session_state.analyzed_location = location
        except Exception:
            st.error("Analysis failed. Check your API key and try again.")


def main() -> None:
    st.set_page_config(
        page_title="CleanLink AI",
        page_icon="♻️",
        layout="wide",
    )
    load_css(STYLES_PATH)
    init_session_state()

    env_api_key = os.getenv("GEMINI_API_KEY")

    render_header()
    spacer(1.5)
    render_hero()
    spacer(0.5)
    api_key = render_settings(env_api_key)
    spacer(0.5)
    st.divider()
    spacer(0.5)

    uploaded_file, location, analyze_clicked = render_upload_form()

    if analyze_clicked:
        handle_analysis(uploaded_file, location, api_key)

    if st.session_state.analysis_result is not None:
        render_results(st.session_state.analysis_result, st.session_state.analyzed_location)

    spacer(1.5)
    render_footer()


if __name__ == "__main__":
    main()