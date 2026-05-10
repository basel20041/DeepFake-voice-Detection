import streamlit as st
import numpy as np
import librosa
import torch
import torch.nn as nn
from scipy import stats
import tempfile
import os
import time
import model

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AudioGuard · Deepfake Detector",
    page_icon="🎙️",
    layout="centered",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark background */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Title area */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #00e5ff 0%, #7b61ff 60%, #ff4b8b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #6b6b8a;
    font-size: 0.95rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}

/* Cards */
.card {
    background: #12121e;
    border: 1px solid #1e1e32;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}

/* Result badges */
.result-real {
    background: linear-gradient(135deg, #00e5ff22, #00e5ff11);
    border: 1px solid #00e5ff55;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    text-align: center;
    color: #00e5ff;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    letter-spacing: 0.08em;
}
.result-fake {
    background: linear-gradient(135deg, #ff4b8b22, #ff4b8b11);
    border: 1px solid #ff4b8b55;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    text-align: center;
    color: #ff4b8b;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    letter-spacing: 0.08em;
}

/* Mono small text */
.mono { font-family: 'Space Mono', monospace; font-size: 0.78rem; color: #5a5a7a; }

/* Slider label tweak */
label { font-family: 'Syne', sans-serif !important; }

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: #0e0e1a !important;
    border: 1px dashed #2a2a45 !important;
    border-radius: 10px !important;
}

/* Progress bar color */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #00e5ff, #7b61ff) !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #00e5ff, #7b61ff);
    color: #0a0a0f;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.06em;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.6rem;
    width: 100%;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; }

/* Sample vote bars */
.vote-bar-container { margin: 0.3rem 0; }
.vote-label { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #9090b0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>AudioGuard</h1>
  <p>DEEPFAKE AUDIO DETECTION · SPECTROGRAM CLASSIFIER</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

SR = 22050  # librosa default sample rate
SEGMENT_DURATION = 1  # seconds per segment

def load_audio(path: str):
    """Load audio with librosa and return (array, sample_rate)."""
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y, sr


def split_into_segments(y: np.ndarray, sr: int, segment_sec: int = SEGMENT_DURATION):
    """Split audio array into fixed-length segments of `segment_sec` seconds."""
    seg_len = sr * segment_sec
    num_segments = len(y) // seg_len
    if num_segments == 0:
        return [y]  # audio shorter than one segment → use whole clip
    segments = []
    for i in range(num_segments):
        segments.append(y[i * seg_len : (i + 1) * seg_len])
    return segments

MODELS_SETTINGS = {
    "Model 1":{
        "n_fft":512,
        "Hop length":128,
        "Mel":True,
        "Normalize":False,
    },
    "Model 2":{
        "n_fft":512,
        "Hop length":128,
        "Mel":False,
        "Normalize":True,
    },
    "Model 3":{
        "n_fft":1024,
        "Hop length":256,
        "Mel":True,
        "Normalize":False,
    },
    "Model 4":{
        "n_fft":1024,
        "Hop length":256,
        "Mel":False,
        "Normalize":True,
    },
    "Model 5":{
        "n_fft":2048,
        "Hop length":512,
        "Mel":True,
        "Normalize":False,
    },
    "Model 6":{
        "n_fft":2048,
        "Hop length":512,
        "Mel":False,
        "Normalize":True,
    },
}
def audio_to_spec(model:str,segment: np.ndarray, sr: int = SR):
    """Convert a 1-D audio array → spectrogram → torch tensor (1, F, T)."""
    settings = MODELS_SETTINGS.get(model)
    if settings['Mel']:
        spec = librosa.feature.melspectrogram(y=segment,n_fft=settings['n_fft'],
                                              hop_length=settings['Hop length'],
                                                 sr=sr,
                                                 n_mels=128,
                                                 fmax=8000)
    else :
        spec = librosa.stft(segment,n_fft=settings['n_fft'],hop_length=settings['Hop length'])
    spec = librosa.power_to_db(spec, ref=np.max)
    if settings['Normalize']:
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, F, T)
    return tensor


def get_output(model,model_name:str, segment: np.ndarray, threshold: float = 0.5, sr: int = SR):
    """
    Run inference on a single segment.
    Returns 'Real' or 'Fake' and the raw sigmoid score.
    """
    tensor = audio_to_spec(model_name,segment, sr)
    tensor = tensor.unsqueeze(0)  # (1, 1, F, T)  batch dim
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        score = output.item()
    label = "Real" if score < threshold else "Fake"
    return label, score


def predict(model,model_name:str, y: np.ndarray, sr: int, max_samples: int = 10, threshold: float = 0.5, seg_sec: int = 1):
    """
    Full pipeline:
      1. Split audio into 1-s segments.
      2. Sample up to `max_samples` evenly spaced segments.
      3. Run inference on each.
      4. Return mode label + per-sample results.
    """
    segments = split_into_segments(y, sr, segment_sec=seg_sec)
    total = len(segments)

    # Evenly spaced indices, capped at max_samples
    num_samples = min(max_samples, total)
    indices = np.linspace(0, total - 1, num_samples, dtype=int)
    chosen = [segments[i] for i in indices]

    results = []
    for seg in chosen:
        label, score = get_output(model,model_name=model_name, segment=seg, threshold=threshold, sr=sr)
        results.append({"label": label, "score": score})

    labels = [r["label"] for r in results]
    final_label = stats.mode(labels, keepdims=True).mode[0]
    return final_label, results, total


# ─────────────────────────────────────────────
#  Sidebar – model config
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("### 🤖 Select Model")
    model_options = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6"]
    selected_model = st.radio("", model_options, index=0, label_visibility="collapsed")

    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01,
                          help="Sigmoid score ≥ threshold → Real, else Fake")
    max_samples = st.slider("Max segments to analyse", 1, 10, 5,
                            help="Up to 10 evenly-spaced 1-second clips")
    # seg_sec = st.slider("Segment length (seconds)", 1, 5, 1)

    st.markdown("---")


# ─────────────────────────────────────────────
#  Placeholder model (replace with your own)
# ─────────────────────────────────────────────


MODEL_PATHS = {
    "Model 1": "Models Best Weights/Model1_best_weight.pth",
    "Model 2": "Models Best Weights/Model2_best_weight.pth",
    "Model 3": "Models Best Weights/Model3_best_weight.pth",
    "Model 4": "Models Best Weights/Model4_best_weight.pth",
    "Model 5": "Models Best Weights/Model5_best_weight.pth",
    "Model 6": "Models Best Weights/Model6_best_weight.pth",
}
MODELS = {
    "Model 1":model.Model1,
    "Model 2":model.Model2,
    "Model 3":model.Model3,
    "Model 4":model.Model4,
    "Model 5":model.Model5,
    "Model 6":model.Model6,
}
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    my_model = MODELS[model_name]()
    path = MODEL_PATHS.get(model_name)
    if path and os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if isinstance(state, dict):
            my_model.load_state_dict(state, strict=False)
    my_model.eval()
    return my_model


# ─────────────────────────────────────────────
#  Main UI
# ─────────────────────────────────────────────

st.markdown('<div class="card">', unsafe_allow_html=True)
audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "ogg", "m4a"])
st.markdown('</div>', unsafe_allow_html=True)

if audio_file:
    st.audio(audio_file)

    run = st.button("🔍 Analyse")

    if run:
        # Load model
        with st.spinner("Loading model…"):
            my_model = load_model(selected_model)

        # Save uploaded audio to temp file
        suffix = os.path.splitext(audio_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file.read())
            tmp_audio_path = tmp.name

        # Load audio
        with st.spinner("Reading audio…"):
            y, sr = load_audio(tmp_audio_path)
        os.unlink(tmp_audio_path)

        duration = len(y) / sr

        # Run inference
        progress = st.progress(0, text="Analysing segments…")
        segments = split_into_segments(y, sr, segment_sec=1)
        total = len(segments)
        num_samples = min(max_samples, total)
        indices = np.linspace(0, total - 1, num_samples, dtype=int)

        per_sample = []
        for step, idx in enumerate(indices):
            label, score = get_output(my_model,model_name=selected_model,segment= segments[idx], threshold=threshold, sr=sr)
            per_sample.append({"label": label, "score": score, "segment": idx})
            progress.progress((step + 1) / num_samples, text=f"Segment {step+1}/{num_samples}")
            time.sleep(0.05)  # visual feedback

        progress.empty()

        labels = [r["label"] for r in per_sample]
        final_label = stats.mode(labels, keepdims=True).mode[0]
        real_count = labels.count("Real")
        fake_count = labels.count("Fake")

        # ── Result ──
        st.markdown("---")
        css_class = "result-real" if final_label == "Real" else "result-fake"
        icon = "✅" if final_label == "Real" else "⚠️"
        st.markdown(f'<div class="{css_class}">{icon} &nbsp; {final_label.upper()}</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <p class="mono" style="text-align:center; margin-top:0.6rem;">
          Duration: {duration:.2f}s &nbsp;|&nbsp; Segments analysed: {num_samples}/{total}
          &nbsp;|&nbsp; Votes — Real: {real_count} · Fake: {fake_count}
        </p>
        """, unsafe_allow_html=True)

        # ── Per-sample breakdown ──
        with st.expander("Per-segment breakdown"):
            cols = st.columns([1, 2, 2])
            cols[0].markdown('<p class="mono">Segment</p>', unsafe_allow_html=True)
            cols[1].markdown('<p class="mono">Label</p>', unsafe_allow_html=True)
            cols[2].markdown('<p class="mono">Sigmoid score</p>', unsafe_allow_html=True)
            for r in per_sample:
                c0, c1, c2 = st.columns([1, 2, 2])
                c0.markdown(f'<p class="mono">#{r["segment"]}</p>', unsafe_allow_html=True)
                color = "#00e5ff" if r["label"] == "Real" else "#ff4b8b"
                c1.markdown(f'<p style="color:{color}; font-family:Space Mono,monospace; font-size:0.8rem;">{r["label"]}</p>', unsafe_allow_html=True)
                c2.progress(float(r["score"]))

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding:2.5rem 1rem;">
      <p style="font-size:2.5rem; margin-bottom:0.5rem;">🎙️</p>
      <p style="color:#3a3a5a; font-family:'Space Mono',monospace; font-size:0.85rem;">
        Upload a WAV, MP3, FLAC, OGG or M4A file to begin analysis.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("""
<p class="mono" style="text-align:center; margin-top:2rem; color:#2a2a3a;">
  AudioGuard · Mel-spectrogram · Mode voting over segments
</p>
""", unsafe_allow_html=True)