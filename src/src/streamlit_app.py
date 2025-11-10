import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import sys
from PIL import Image
from torchvision import transforms
from pathlib import Path
import yaml
import json



# Ensure project root and src/ are on sys.path (works before and after flattening)
def _discover_project_root_and_src():
    here = Path(__file__).resolve()
    src_dir = None
    for anc in here.parents:
        if anc.name == "src":
            src_dir = anc
            project_root = anc.parent
            break
    if src_dir is None:
        # Fallback to original layout assumption
        project_root = here.parents[1]
        src_dir = project_root / "src"
    return project_root, src_dir

PROJECT_ROOT, SRC_DIR = _discover_project_root_and_src()
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import from src.<module>
from src.infer import InferenceModel
from src.gemini import analyze_plant_with_gemini
# YOLO fully removed; no detector UI or messaging

st.set_page_config(page_title="Multi-Crop Recognition", layout="wide")

# Theming & animated background
import streamlit as st

# Enhanced professional green theme style for Streamlit
professional_green_theme = """
<style>
/* Root color variables */
:root {
  --primary: #1b5e20;
  --accent: #4caf50;
  --bg-start: #0c1a0c;
  --bg-end: #142914;
  --card-bg: #1e3c16;
  --text-color: #e3f2df;
  --header-color: #a9d18e;
  --button-bg: #4caf50;
  --button-hover-bg: #6bbf59;
  --shadow-color: rgba(27, 94, 32, 0.4);
}

/* Overall app background and text */
.stApp {
  background: linear-gradient(135deg, var(--bg-start), var(--bg-end));
  color: var(--text-color);
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  line-height: 1.6;
  padding: 1rem;
}

/* Animated soft gradient overlay */
.bg-anim {
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
  z-index: -1;
  background:
    radial-gradient(1200px 600px at 10% 20%, rgba(76,175,80,0.1), transparent),
    radial-gradient(900px 500px at 90% 10%, rgba(27,94,32,0.1), transparent),
    radial-gradient(1000px 700px at 50% 90%, rgba(56,142,60,0.08), transparent);
  animation: floatbg 15s ease-in-out infinite alternate;
}
@keyframes floatbg {
  from { transform: translateY(-8px); }
  to { transform: translateY(8px); }
}

/* Sidebar style */
div[data-testid="stSidebar"] {
  background-color: var(--card-bg);
  box-shadow: 2px 0 8px var(--shadow-color);
  border-radius: 0 10px 10px 0;
  padding: 1rem 1.5rem;
}

/* Text elements */
.stMarkdown, .stText, .stAlert, .stDataFrame {
  color: var(--text-color);
  font-weight: 400;
}

/* Headers with subtle shadows */
h1, h2, h3 {
  color: var(--header-color);
  text-shadow: 1px 1px 3px var(--shadow-color);
  margin-bottom: 0.5rem;
}

/* Buttons with smooth transitions and shadows */
button[kind="primary"], .stButton>button {
  background-color: var(--button-bg);
  color: #0b150c;
  border: none;
  padding: 0.5rem 1.2rem;
  border-radius: 5px;
  font-weight: 600;
  box-shadow: 0 3px 8px var(--shadow-color);
  transition: background-color 0.3s ease, box-shadow 0.3s ease;
  cursor: pointer;
}
button[kind="primary"]:hover, .stButton>button:hover {
  background-color: var(--button-hover-bg);
  box-shadow: 0 5px 15px var(--shadow-color);
}

/* Inputs and text areas styling */
input, textarea, select {
  background-color: var(--card-bg);
  border: 1px solid #2f5d2f;
  color: var(--text-color);
  border-radius: 5px;
  padding: 0.4rem 0.7rem;
  font-size: 1rem;
  transition: border-color 0.3s ease;
}
input:focus, textarea:focus, select:focus {
  border-color: var(--accent);
  outline: none;
  box-shadow: 0 0 8px var(--accent);
}

/* Tables styling */
.stDataFrame table {
  border-collapse: collapse;
  width: 100%;
  background: var(--card-bg);
  color: var(--text-color);
  border-radius: 8px;
  overflow: hidden;
}
.stDataFrame th, .stDataFrame td {
  padding: 0.5rem 1rem;
  border-bottom: 1px solid #2f5d2f;
  text-align: left;
}
.stDataFrame th {
  background-color: var(--primary);
  color: #d4eecd;
}

/* Hide default Streamlit footer */
footer {
  visibility: hidden;
  height: 0;
  margin: 0;
  padding: 0;
}
</style>
<div class="bg-anim"></div>
"""

with open("configs/train.yaml", "r") as f:
    cfg = yaml.safe_load(f)
 

st.markdown(professional_green_theme, unsafe_allow_html=True)
with open("reports/labels.json", "r") as f:
    labels = json.load(f)

predicted_index = 5  # example predicted class index as int
predicted_label = labels[str(predicted_index)]


@st.cache_resource
def load_model():
    model = InferenceModel("weights/best.pt", model_name=None, device=None)
    return model

model = load_model()

st.title("🌱 Multi-Crop Plant Recognition")
from src import gemini as _gemini

# Detect Gemini keys silently; if none, hide Gemini section
try:
    _keys = _gemini._get_all_keys(None)
except Exception:
    _keys = []
HAS_GEMINI = bool(_keys)
analysis_mode = st.selectbox(
    "Analysis mode",
    options=["Fast (3–5s)", "Detailed (8–12s)"],
    index=1,
    help="Fast mode returns concise guidance quickly. Detailed mode provides longer analysis."
)
left, right = st.columns([2,1])
with left:
    uploaded = st.file_uploader("Upload a plant image", type=["jpg","jpeg","png"])
with right:
    cam = st.camera_input("Or capture from camera")

img = None
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
elif cam is not None:
    img = Image.open(cam).convert("RGB")

if img:
    # Show input image as requeste
    st.image(img, caption="Input", width='content')
    rgb = np.array(img)
    # Full-image classification (YOLO removed)
    crop_rgb = rgb
    context_note = ""
    # Use multi-crop for robustness (works without YOLO training too)
    preds = model.predict_multicrop(crop_rgb, topk=5, tta="fast")
    top1_class, top1_conf = preds[0]  # Get first prediction (class, confidence)
    # If low confidence, retry with stronger TTA to reduce false predictions
    if top1_conf < 0.35:
        preds = model.predict_multicrop(crop_rgb, topk=5, tta="full")
        top1_class, top1_conf = preds[0]


    trained_preds = model.predict_multicrop(rgb, topk=5, tta="full")
    top_class, top_conf = trained_preds[0]
    st.markdown("### Trained Model Prediction")
    st.markdown(f"**Top prediction:** {top_class} ({top_conf:.2f})")
    st.markdown("**Top-5:**")
    for label, conf in trained_preds:
        st.write(f"- {label}: {conf*100:.1f}%")

    # Gemini analysis section (only if key available)
    if HAS_GEMINI:
        st.divider()
        st.subheader("Gemini Disease Analysis & Remedies")
        try:
            # Adaptive hand-off: compute margin and entropy to decide how to brief Gemini
            probs = np.array([p for _, p in preds], dtype=float)
            p1 = float(probs[0]) if len(probs) else 0.0
            p2 = float(probs[1]) if len(probs) > 1 else 0.0
            margin = p1 - p2
            # normalized entropy (0..1)
            eps = 1e-8
            norm = probs.sum() + eps
            q = probs / norm
            entropy = float(-(q * np.log(q + eps)).sum() / (np.log(len(q) + eps))) if len(q) > 1 else 0.0

            # Decide confidence bucket and build candidate_str
            # High confidence if p1>=0.70 and margin>=0.20 and entropy<=0.65
            # Medium if p1>=0.50 or margin>=0.10
            # Low otherwise
            if (p1 >= 0.70 and margin >= 0.20 and entropy <= 0.65):
                conf_bucket = "High"
                candidate_str = top1_class
            elif (p1 >= 0.50 or margin >= 0.10):
                conf_bucket = "Medium"
                k = min(3, len(preds))
                topk = ", ".join([f"{c} ({p:.2f})" for c, p in preds[:k]])
                candidate_str = f"Candidates: {topk}"
            else:
                conf_bucket = "Low"
                k = min(5, len(preds))
                topk = ", ".join([f"{c} ({p:.2f})" for c, p in preds[:k]])
                candidate_str = f"Uncertain; please identify species from image. Candidate list: {topk}"

            st.caption(f"Model confidence: {conf_bucket} | p1={p1:.2f}, margin={margin:.2f}, entropy={entropy:.2f}")
            # Look-alike nudge: cucurbits vs brassicas to reduce confusion
            cucurbits = {"Pumpkins, squash and gourds plant", "Cucumber plant", "Zucchini plant", "Pumpkin plant", "Squash plant"}
            brassicas = {"Cabbages and other brassicas plant", "Mustard greens plant", "Rapeseed (Canola) plant", "Brussels sprout plant"}
            top_names = {c for c,_ in preds[:5]}
            if top_names & cucurbits and (top_names & brassicas):
                candidate_str += ", Possible species group: cucurbits (pumpkin/squash/cucumber)"
            # Parameters based on mode
            if analysis_mode.startswith("Fast"):
                timeout_sec = 4.0
                max_tokens = 320
                max_side = 640
                spinner_msg = "Analyzing with Gemini (fast mode)..."
                model_name = None  # use default fast model
            else:
                timeout_sec = 12.0
                max_tokens = 1200
                max_side = 1024
                spinner_msg = "Analyzing with Gemini (detailed mode)..."
                model_name = "gemini-1.5-pro"

            with st.spinner(spinner_msg):
                # If we cropped a region for fruit context, send that crop to Gemini for visual grounding
                img_for_llm = Image.fromarray(crop_rgb)
                species_text = candidate_str if not context_note else f"{context_note}. {candidate_str}"
                result = analyze_plant_with_gemini(
                    img_for_llm,
                    species_text,
                    top1_conf,
                    api_key=None,
                    timeout_sec=timeout_sec,
                    max_output_tokens=max_tokens,
                    max_side=max_side,
                    temperature=0.7 if analysis_mode.startswith("Detailed") else 0.6,
                    model_name=model_name,
                )
            content = result.get("gemini_output", "No response")
            # Language toggle and sectioned display
            st.divider()
            lang = st.radio("Language / भाषा", ["English", "Hindi"], index=0, horizontal=True)

            def extract_section(text: str, header: str) -> str:
                import re
                # Capture from header to next top-level header or end
                pattern = rf"(?s)###\s+{re.escape(header)}\s*(.*?)(?=\n###\s+|\Z)"
                m = re.search(pattern, text)
                return m.group(1).strip() if m else ""

            eng = extract_section(content, "English")
            hin = extract_section(content, "Hindi")
            if lang == "English" and eng:
                st.markdown("### English\n" + eng)
            elif lang == "Hindi" and hin:
                st.markdown("### Hindi\n" + hin)
            else:
                # Fallback: show full content if parsing failed
                st.markdown(content)
            if "latency_sec" in result:
                st.caption(f"Gemini latency: {result['latency_sec']}s | Mode: {analysis_mode}")
                # Auto-upgrade: if fast mode produced fallback or too-short output, run a detailed retry automatically
                if analysis_mode.startswith("Fast") and ("Response timed out" in content or len(content) < 300):
                    with st.spinner("Generating detailed analysis (one-time upgrade)..."):
                        result2 = analyze_plant_with_gemini(
                            img,
                            top1_class,
                            top1_conf,
                            api_key=None,
                            timeout_sec=9.0,
                            max_output_tokens=900,
                            max_side=896,
                            temperature=0.7,
                        )
                    st.markdown("---")
                    st.subheader("Detailed Analysis")
                    st.markdown(result2.get("gemini_output", "No response"))
                    if "latency_sec" in result2:
                        st.caption(f"Detailed latency: {result2['latency_sec']}s")
                # Detailed retry: if Detailed mode still times out or is too short, force a single extended retry
                if analysis_mode.startswith("Detailed") and ("Response timed out" in content or len(content) < 600):
                    with st.spinner("Retrying with extended detailed mode..."):
                        result3 = analyze_plant_with_gemini(
                            img,
                            top1_class,
                            top1_conf,
                            api_key=None,
                            timeout_sec=18.0,
                            max_output_tokens=1600,
                            max_side=1280,
                            temperature=0.7,
                            model_name="gemini-1.5-pro",
                        )
                    st.markdown("---")
                    st.subheader("Extended Detailed Analysis")
                    st.markdown(result3.get("gemini_output", "No response"))
                    if "latency_sec" in result3:
                        st.caption(f"Extended detailed latency: {result3['latency_sec']}s")
        except Exception as e:
            st.error(f"Gemini analysis failed: {e}")
