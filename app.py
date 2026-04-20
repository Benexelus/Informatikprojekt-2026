import streamlit as st
import numpy as np
import json
import os
import time
from PIL import Image
from datetime import datetime
import glob

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trash Monitor",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.alert-box {
    background: #fff3cd;
    border: 2px solid #ff4b4b;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%   { border-color: #ff4b4b; }
    50%  { border-color: #ffaa00; }
    100% { border-color: #ff4b4b; }
}
.full-alert {
    background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
    border: 3px solid #ff4b4b;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.status-ok {
    background: #d4edda;
    border: 1px solid #28a745;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    color: #155724;
    font-weight: bold;
}
.status-full {
    background: #f8d7da;
    border: 1px solid #dc3545;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    color: #721c24;
    font-weight: bold;
}
.cam-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.cam-card-full {
    border: 3px solid #ff4b4b !important;
    box-shadow: 0 0 15px #ff4b4b55;
}
</style>
""", unsafe_allow_html=True)

# ── Sound alert (HTML audio) ──────────────────────────────────────────────────
ALERT_SOUND = """
<audio id="alertSound" autoplay>
  <source src="https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3" type="audio/mpeg">
</audio>
<script>
  var audio = document.getElementById('alertSound');
  audio.volume = 0.5;
  audio.play().catch(e => console.log('Autoplay blocked:', e));
</script>
"""

# ── Teachable Machine model loader ───────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load Teachable Machine Keras model + labels."""
    model_path = "model/keras_model.h5"
    labels_path = "model/labels.txt"

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None, None

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
        with open(labels_path, "r") as f:
            labels = [line.strip().split(" ", 1)[-1] for line in f.readlines()]
        return model, labels
    except Exception as e:
        st.error(f"Modell konnte nicht geladen werden: {e}")
        return None, None


def predict_image(model, labels, img: Image.Image):
    """Run inference on a PIL image. Returns (label, confidence)."""
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return labels[idx], float(preds[idx])


def is_full(label: str) -> bool:
    """Returns True if the label indicates a full bin."""
    return "voll" in label.lower() or "full" in label.lower()


# ── Camera helpers ────────────────────────────────────────────────────────────
def get_latest_image(cam_folder: str):
    """Return the most recently modified image in a folder."""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(cam_folder, p)))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def get_camera_folders():
    """List all cam_* folders in camera_images/."""
    base = "camera_images"
    if not os.path.exists(base):
        os.makedirs(base)
    folders = sorted([
        os.path.join(base, d)
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ])
    return folders


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🗑️ Trash Monitor")
    st.markdown("---")

    st.subheader("Modell laden")
    uploaded_model = st.file_uploader("keras_model.h5", type=["h5"])
    uploaded_labels = st.file_uploader("labels.txt", type=["txt"])

    if uploaded_model and uploaded_labels:
        os.makedirs("model", exist_ok=True)
        with open("model/keras_model.h5", "wb") as f:
            f.write(uploaded_model.read())
        with open("model/labels.txt", "w") as f:
            f.write(uploaded_labels.read().decode("utf-8"))
        st.success("Modell gespeichert! Bitte Seite neu laden.")
        st.cache_resource.clear()

    st.markdown("---")
    st.subheader("Einstellungen")
    threshold = st.slider("Schwellenwert (Konfidenz)", 0.5, 1.0, 0.75, 0.05,
                          help="Ab welcher Konfidenz gilt der Eimer als voll?")
    refresh_interval = st.selectbox("Auto-Refresh", [10, 30, 60, 120, 300],
                                    index=1, format_func=lambda x: f"alle {x} Sekunden")

    st.markdown("---")
    st.subheader("Kamera-Ordner")
    cam_folders = get_camera_folders()
    if cam_folders:
        for f in cam_folders:
            st.code(f, language=None)
    else:
        st.info("Noch keine Ordner in camera_images/")

    new_cam = st.text_input("Neuen Kamera-Ordner anlegen")
    if st.button("Ordner erstellen") and new_cam:
        path = os.path.join("camera_images", new_cam.strip())
        os.makedirs(path, exist_ok=True)
        st.success(f"Ordner '{path}' erstellt!")
        st.rerun()

    st.markdown("---")
    if st.button("🔄 Jetzt aktualisieren"):
        st.rerun()

# ── Load model ────────────────────────────────────────────────────────────────
model, labels = load_model()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🗑️ Mülleimer-Überwachung")

if model is None:
    st.warning(
        "⚠️ Kein Modell geladen. Bitte lade links **keras_model.h5** und **labels.txt** "
        "aus deinem Teachable Machine Export hoch."
    )
    st.info(
        "**So exportierst du dein Teachable Machine Modell:**\n"
        "1. Gehe zu [teachablemachine.withgoogle.com](https://teachablemachine.withgoogle.com)\n"
        "2. Trainiere dein Modell mit den Klassen **'voll'** und **'nicht voll'**\n"
        "3. Klicke auf **Modell exportieren → Tensorflow → Keras**\n"
        "4. Lade `keras_model.h5` und `labels.txt` hier hoch"
    )

cam_folders = get_camera_folders()

if not cam_folders:
    st.info("📁 Noch keine Kamera-Ordner vorhanden. Erstelle einen im Sidebar.")
    st.stop()

# ── Analyse all cameras ───────────────────────────────────────────────────────
results = []
any_full = False

for folder in cam_folders:
    cam_name = os.path.basename(folder)
    img_path = get_latest_image(folder)

    entry = {
        "name": cam_name,
        "folder": folder,
        "img_path": img_path,
        "label": None,
        "confidence": None,
        "full": False,
        "timestamp": None,
    }

    if img_path:
        entry["timestamp"] = datetime.fromtimestamp(os.path.getmtime(img_path)).strftime("%d.%m.%Y %H:%M:%S")
        if model:
            img = Image.open(img_path)
            label, conf = predict_image(model, labels, img)
            entry["label"] = label
            entry["confidence"] = conf
            entry["full"] = is_full(label) and conf >= threshold

    if entry["full"]:
        any_full = True

    results.append(entry)

# ── Global alert ─────────────────────────────────────────────────────────────
full_cams = [r for r in results if r["full"]]

if full_cams:
    st.markdown(
        f"""<div class="alert-box">
        <h2>🚨 ACHTUNG: {len(full_cams)} Mülleimer überfüllt!</h2>
        <p>{'  |  '.join(['📍 ' + r['name'] for r in full_cams])}</p>
        </div>""",
        unsafe_allow_html=True
    )
    st.components.v1.html(ALERT_SOUND, height=0)

# ── Full bins — prominent display ─────────────────────────────────────────────
if full_cams:
    st.markdown("## 🔴 Überfüllte Eimer")
    cols = st.columns(min(len(full_cams), 3))
    for i, r in enumerate(full_cams):
        with cols[i % 3]:
            st.markdown('<div class="full-alert">', unsafe_allow_html=True)
            st.markdown(f"### 🗑️ {r['name']}")
            if r["img_path"]:
                st.image(r["img_path"], use_column_width=True)
            st.markdown(
                f"**{r['label']}** — {r['confidence']*100:.1f}% Konfidenz<br>"
                f"🕐 {r['timestamp']}",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

# ── All cameras overview ──────────────────────────────────────────────────────
st.markdown("## 📷 Alle Kameras")

cols = st.columns(min(len(results), 3))
for i, r in enumerate(results):
    with cols[i % 3]:
        border_class = "cam-card-full" if r["full"] else ""
        st.markdown(f'<div class="cam-card {border_class}">', unsafe_allow_html=True)
        st.markdown(f"**📍 {r['name']}**")

        if r["img_path"]:
            st.image(r["img_path"], use_column_width=True)

            if r["label"]:
                status_class = "status-full" if r["full"] else "status-ok"
                icon = "🔴" if r["full"] else "🟢"
                st.markdown(
                    f'<div class="{status_class}">{icon} {r["label"]} ({r["confidence"]*100:.1f}%)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("Kein Modell — Bild vorhanden")

            st.caption(f"🕐 Letztes Bild: {r['timestamp']}")
        else:
            st.warning("Noch kein Bild im Ordner")

        st.markdown("</div>", unsafe_allow_html=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"🔄 Automatische Aktualisierung alle {refresh_interval} Sekunden | Letztes Update: {datetime.now().strftime('%H:%M:%S')}")
time.sleep(refresh_interval)
st.rerun()
