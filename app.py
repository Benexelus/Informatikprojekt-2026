import streamlit as st
import numpy as np
import os
import base64
import requests
import json
from PIL import Image
from datetime import datetime
from io import BytesIO

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trash Monitor",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.alert-banner {
    background: #fff0f0;
    border: 2px solid #ff4b4b;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
}
.full-card {
    border: 3px solid #ff4b4b !important;
    border-radius: 12px;
    padding: 1rem;
    background: #fff5f5;
    box-shadow: 0 0 18px #ff4b4b44;
    margin-bottom: 1rem;
}
.ok-card {
    border: 1px solid #28a745;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.no-img-card {
    border: 1px solid #ccc;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    background: #fafafa;
}
.status-full  { color: #dc3545; font-weight: bold; font-size: 1.1rem; }
.status-ok    { color: #28a745; font-weight: bold; font-size: 1.1rem; }
.status-none  { color: #888;    font-style: italic; }
.cam-label    { font-size: 1rem; font-weight: 600; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ── GitHub helpers ────────────────────────────────────────────────────────────
GITHUB_TOKEN  = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO   = st.secrets.get("GITHUB_REPO", "")   # "username/reponame"
IMAGES_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
IMAGES_FOLDER = "camera_images"                       # folder inside repo
CAMERAS_FILE  = "cameras.json"                        # camera registry in repo

def gh_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

def gh_get_file(path: str):
    """Fetch a file from GitHub. Returns (content_bytes, sha) or (None, None)."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}?ref={IMAGES_BRANCH}"
    r = requests.get(url, headers=gh_headers(), timeout=10)
    if r.status_code == 200:
        data = r.json()
        content = base64.b64decode(data["content"])
        return content, data["sha"]
    return None, None

def gh_put_file(path: str, content_bytes: bytes, sha: str | None, message: str):
    """Create or update a file on GitHub."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode(),
        "branch": IMAGES_BRANCH,
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=gh_headers(), json=payload, timeout=15)
    return r.status_code in (200, 201)

def load_cameras() -> dict:
    """Load camera registry from GitHub. Returns dict {cam_id: {name, connection, ...}}."""
    content, _ = gh_get_file(CAMERAS_FILE)
    if content:
        return json.loads(content.decode())
    return {}

def save_cameras(cameras: dict):
    """Save camera registry to GitHub."""
    _, sha = gh_get_file(CAMERAS_FILE)
    gh_put_file(
        CAMERAS_FILE,
        json.dumps(cameras, indent=2).encode(),
        sha,
        "Update camera registry"
    )

def save_image_to_github(cam_id: str, img_bytes: bytes, ext: str = "jpg") -> bool:
    """Overwrite the single latest image for a camera on GitHub."""
    path = f"{IMAGES_FOLDER}/{cam_id}/latest.{ext}"
    _, sha = gh_get_file(path)
    return gh_put_file(path, img_bytes, sha, f"Update image for {cam_id}")

def load_image_from_github(cam_id: str) -> Image.Image | None:
    """Load the latest image for a camera from GitHub."""
    for ext in ("jpg", "jpeg", "png"):
        path = f"{IMAGES_FOLDER}/{cam_id}/latest.{ext}"
        content, _ = gh_get_file(path)
        if content:
            return Image.open(BytesIO(content))
    return None

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model/keras_model.h5") or not os.path.exists("model/labels.txt"):
        return None, None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model("model/keras_model.h5", compile=False)
        with open("model/labels.txt") as f:
            labels = [l.strip().split(" ", 1)[-1] for l in f.readlines()]
        return model, labels
    except Exception as e:
        st.error(f"Modell-Fehler: {e}")
        return None, None

def predict(model, labels, img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return labels[idx], float(preds[idx])

def is_full(label: str) -> bool:
    return "voll" in label.lower() or "full" in label.lower()

# ── Session state defaults ────────────────────────────────────────────────────
if "cameras" not in st.session_state:
    if GITHUB_TOKEN and GITHUB_REPO:
        st.session_state.cameras = load_cameras()
    else:
        st.session_state.cameras = {}

if "selected_cam" not in st.session_state:
    st.session_state.selected_cam = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🗑️ Trash Monitor")
    st.markdown("---")

    github_ok = bool(GITHUB_TOKEN and GITHUB_REPO)
    if github_ok:
        st.success(f"✅ GitHub verbunden\n\n`{GITHUB_REPO}`")
    else:
        st.warning("⚠️ GitHub nicht konfiguriert.\nBitte Secrets setzen (siehe README).")

    st.markdown("---")
    model, labels = load_model()
    if model:
        st.success("✅ KI-Modell geladen")
    else:
        st.error("❌ Kein Modell gefunden\n\n`model/keras_model.h5` fehlt im Repo.")

    st.markdown("---")
    threshold = st.slider("Konfidenz-Schwellenwert", 0.5, 1.0, 0.75, 0.05,
                          help="Ab wann gilt ein Eimer als voll?")

    st.markdown("---")
    if st.button("🔄 Aktualisieren"):
        st.session_state.cameras = load_cameras()
        st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_monitor, tab_cameras, tab_test = st.tabs([
    "📺 Monitor", "📷 Kameras verwalten", "🧪 Test-Upload"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — MONITOR
# ════════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    cameras = st.session_state.cameras

    if not cameras:
        st.info("Noch keine Kameras verbunden. Gehe zu **Kameras verwalten**.")
        st.stop()

    # ── Analyse all cameras ──────────────────────────────────────────────────
    results = []
    for cam_id, cam_info in cameras.items():
        img = load_image_from_github(cam_id) if github_ok else None
        label, conf, full = None, None, False
        if img and model:
            label, conf = predict(model, labels, img)
            full = is_full(label) and conf >= threshold
        results.append({
            "id": cam_id,
            "name": cam_info.get("name", cam_id),
            "connection": cam_info.get("connection", ""),
            "img": img,
            "label": label,
            "conf": conf,
            "full": full,
        })

    full_cams = [r for r in results if r["full"]]

    # ── Global alert ─────────────────────────────────────────────────────────
    if full_cams:
        names = " | ".join(["📍 " + r["name"] for r in full_cams])
        st.markdown(
            f'<div class="alert-banner"><h2>🚨 {len(full_cams)} Mülleimer überfüllt!</h2><p>{names}</p></div>',
            unsafe_allow_html=True
        )
        # Sound
        st.components.v1.html("""
        <script>
        const ctx = new AudioContext();
        function beep(freq, dur, vol) {
            const o = ctx.createOscillator();
            const g = ctx.createGain();
            o.connect(g); g.connect(ctx.destination);
            o.frequency.value = freq;
            g.gain.setValueAtTime(vol, ctx.currentTime);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur);
            o.start(ctx.currentTime);
            o.stop(ctx.currentTime + dur);
        }
        beep(880, 0.3, 0.4);
        setTimeout(() => beep(880, 0.3, 0.4), 400);
        setTimeout(() => beep(1100, 0.5, 0.5), 800);
        </script>
        """, height=0)

    # ── Überfüllte Eimer oben groß ────────────────────────────────────────────
    if full_cams:
        st.markdown("## 🔴 Überfüllte Eimer")
        cols = st.columns(min(len(full_cams), 3))
        for i, r in enumerate(full_cams):
            with cols[i % 3]:
                st.markdown('<div class="full-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="cam-label">🗑️ {r["name"]}</div>', unsafe_allow_html=True)
                if r["img"]:
                    st.image(r["img"], use_column_width=True)
                st.markdown(f'<div class="status-full">🔴 {r["label"]} — {r["conf"]*100:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

    # ── Alle Kameras ──────────────────────────────────────────────────────────
    st.markdown("## 📷 Alle Kameras")
    cols = st.columns(min(len(results), 3))
    for i, r in enumerate(results):
        with cols[i % 3]:
            card_class = "full-card" if r["full"] else ("ok-card" if r["img"] else "no-img-card")
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            st.markdown(f'<div class="cam-label">📍 {r["name"]}</div>', unsafe_allow_html=True)
            st.caption(f'Verbindung: {r["connection"]}')

            if r["img"]:
                # Clickable image → detail view
                if st.button(f"🔍 Letztes Bild anzeigen", key=f"view_{r['id']}"):
                    st.session_state.selected_cam = r["id"]

                st.image(r["img"], use_column_width=True)

                if r["label"]:
                    icon = "🔴" if r["full"] else "🟢"
                    status_class = "status-full" if r["full"] else "status-ok"
                    st.markdown(
                        f'<div class="{status_class}">{icon} {r["label"]} ({r["conf"]*100:.1f}%)</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown('<div class="status-none">Kein Modell geladen</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-none">⏳ Noch kein Bild empfangen</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # ── Detail-Popup ──────────────────────────────────────────────────────────
    if st.session_state.selected_cam:
        cam_id = st.session_state.selected_cam
        cam_info = cameras.get(cam_id, {})
        img = load_image_from_github(cam_id)

        with st.expander(f"🔍 Detail: {cam_info.get('name', cam_id)}", expanded=True):
            if img:
                st.image(img, caption=f"Letztes Bild von {cam_info.get('name', cam_id)}", use_column_width=True)
                if model:
                    label, conf = predict(model, labels, img)
                    full = is_full(label) and conf >= threshold
                    icon = "🔴" if full else "🟢"
                    st.metric("Ergebnis", f"{icon} {label}", f"{conf*100:.1f}% Konfidenz")
            else:
                st.warning("Kein Bild verfügbar.")
            if st.button("Schließen"):
                st.session_state.selected_cam = None
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — KAMERAS VERWALTEN
# ════════════════════════════════════════════════════════════════════════════════
with tab_cameras:
    st.header("📷 Kameras verwalten")

    # ── Add camera ────────────────────────────────────────────────────────────
    with st.expander("➕ Neue Kamera hinzufügen", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("Kamera-Name", placeholder="z.B. Eingang Rathaus")
            new_id   = st.text_input("Kamera-ID (keine Leerzeichen)", placeholder="z.B. cam_rathaus")
        with col2:
            conn_type = st.selectbox("Verbindungstyp", [
                "HTTP-Server (Raspberry Pi / PC)",
                "WLAN / lokales Netzwerk",
                "LTE / Mobilfunk",
                "Bluetooth (mit Relay-Skript)",
                "Manueller Upload",
            ])
            conn_note = st.text_input("Verbindungsdetail (optional)", placeholder="z.B. IP-Adresse, Gerätename")

        if st.button("Kamera hinzufügen"):
            if not new_name or not new_id:
                st.error("Name und ID sind Pflichtfelder.")
            elif new_id in st.session_state.cameras:
                st.error(f"ID '{new_id}' existiert bereits.")
            else:
                st.session_state.cameras[new_id] = {
                    "name": new_name,
                    "connection": f"{conn_type} — {conn_note}" if conn_note else conn_type,
                    "added": datetime.now().isoformat(),
                }
                if github_ok:
                    save_cameras(st.session_state.cameras)
                st.success(f"Kamera '{new_name}' hinzugefügt!")
                st.rerun()

    st.markdown("---")

    # ── Connection guide ──────────────────────────────────────────────────────
    with st.expander("📡 Verbindungsmöglichkeiten — Anleitung"):
        st.markdown("""
### 1. HTTP-Server (empfohlen für Raspberry Pi / PC)
Die Kamera läuft als Python-Skript und sendet Bilder direkt an die App.

```python
# capture_and_upload.py  (auf dem Raspberry Pi ausführen)
import cv2, requests, time, base64
from datetime import datetime

CAM_ID    = "cam_rathaus"
APP_URL   = "https://DEINE-APP.streamlit.app/upload"   # falls API aktiviert
INTERVAL  = 900   # 15 Minuten

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        _, buf = cv2.imencode(".jpg", frame)
        # Alternativ: direkt in GitHub-Ordner kopieren (bei gemeinsamem Netzwerk)
        with open(f"camera_images/{CAM_ID}/latest.jpg", "wb") as f:
            f.write(buf.tobytes())
    time.sleep(INTERVAL)
```

### 2. WLAN / Netzwerklaufwerk
Kamera-PC und App-Server im selben Netzwerk → Bilder direkt in geteilten Ordner schreiben.

### 3. LTE / Mobilfunk (ESP32-CAM)
ESP32-CAM sendet Bild per HTTP POST an einen kleinen Flask-Server, der das Bild dann auf GitHub pusht.

```python
# relay_server.py  (auf einem Server mit fester IP)
from flask import Flask, request
import requests, base64, json

app = Flask(__name__)
GITHUB_TOKEN = "..."
GITHUB_REPO  = "username/trash-monitor"

@app.route("/upload/<cam_id>", methods=["POST"])
def upload(cam_id):
    img_bytes = request.data
    # Push to GitHub
    path = f"camera_images/{cam_id}/latest.jpg"
    r = requests.get(f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}",
                     headers={"Authorization": f"token {GITHUB_TOKEN}"})
    sha = r.json().get("sha") if r.status_code == 200 else None
    payload = {"message": f"Update {cam_id}", "content": base64.b64encode(img_bytes).decode(), "branch": "main"}
    if sha: payload["sha"] = sha
    requests.put(f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}",
                 headers={"Authorization": f"token {GITHUB_TOKEN}"}, json=payload)
    return "ok"

app.run(host="0.0.0.0", port=5000)
```

### 4. Bluetooth (mit Relay-Skript)
Bluetooth allein reicht nicht für Internetzugang. Lösung: Smartphone als Relay.
- ESP32-CAM → Bluetooth → Python-Skript auf Smartphone (z.B. via Termux) → GitHub API

### 5. Manueller Upload
Bild in der **Test-Upload**-Sektion hochladen — gut für Tests oder als Notfalloption.
        """)

    st.markdown("---")

    # ── Existing cameras ──────────────────────────────────────────────────────
    st.subheader("Verbundene Kameras")
    cameras = st.session_state.cameras
    if not cameras:
        st.info("Noch keine Kameras hinzugefügt.")
    else:
        for cam_id, cam_info in list(cameras.items()):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**{cam_info['name']}**  \n`{cam_id}`")
            with col2:
                st.caption(cam_info.get("connection", ""))
            with col3:
                if st.button("🗑 Entfernen", key=f"del_{cam_id}"):
                    del st.session_state.cameras[cam_id]
                    if github_ok:
                        save_cameras(st.session_state.cameras)
                    st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — TEST UPLOAD
# ════════════════════════════════════════════════════════════════════════════════
with tab_test:
    st.header("🧪 Test-Upload")
    st.info("Lade ein Bild hoch, um den Füllstand eines Mülleimers zu testen.")

    # ── Bild-Upload ──────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Wähle ein Bild aus (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="test_uploader"
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Hochgeladenes Bild", use_column_width=True)

        # ── KI-Analyse ───────────────────────────────────────────────────────
        if model:
            label, conf = predict(model, labels, img)
            full = is_full(label) and conf >= threshold
            icon = "🔴" if full else "🟢"
            
            st.markdown("### 🔍 Analyseergebnis")
            st.metric("Erkennung", f"{icon} {label}", f"{conf*100:.1f}% Konfidenz")
            
            if full:
                st.error("⚠️ Mülleimer ist **voll**!")
            else:
                st.success("✅ Mülleimer ist **nicht voll**.")
        else:
            st.warning("Kein Modell geladen – Analyse nicht möglich.")

        # ── Optional: Speichern auf GitHub ───────────────────────────────────
        if github_ok and GITHUB_REPO:
            st.markdown("---")
            st.markdown("### 📤 Auf GitHub speichern")
            
            cameras = st.session_state.cameras
            if cameras:
                cam_options = {v["name"]: k for k, v in cameras.items()}
                selected_name = st.selectbox(
                    "Kamera auswählen (für Speicherort)",
                    list(cam_options.keys())
                )
                selected_id = cam_options[selected_name]

                if st.button("Bild auf GitHub speichern"):
                    buf = BytesIO()
                    img.save(buf, format="JPEG")
                    ok = save_image_to_github(selected_id, buf.getvalue(), "jpg")
                    if ok:
                        st.success("✅ Bild erfolgreich gespeichert!")
                    else:
                        st.error("❌ Fehler beim Speichern auf GitHub.")
            else:
                st.warning("Keine Kameras definiert – bitte zuerst unter **Kameras verwalten** anlegen.")
tab_monitor, tab_cameras, tab_test, tab_quicktest = st.tabs([
       "📺 Monitor", "📷 Kameras", "📤 Test-Upload", "🔍 Schnelltest"
   ])
# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCHNELLTEST (VERBESSERT)
# ════════════════════════════════════════════════════════════════════════════════
tab_quicktest = st.tabs(["📺 Monitor", "📷 Kameras", "📤 Test-Upload", "🔍 Schnelltest"])[3]  # Index 3 = 4. Reiter

with tab_quicktest:
    st.header("🔍 Schnelltest")
    st.markdown("""
    <style>
    .quicktest-card {
        border: 2px dashed #4b8df8 !important;
        border-radius: 12px;
        padding: 1rem;
        background: #f5f9ff;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="quicktest-card">
        <b>Lade ein Bild hoch für eine sofortige Analyse:</b><br>
        Keine Kamera oder GitHub-Verbindung erforderlich.
    </div>
    """, unsafe_allow_html=True)

    # Bild-Upload
    uploaded_img = st.file_uploader(
        "Bild auswählen (JPG/PNG)", 
        type=["jpg", "jpeg", "png"],
        key="quick_upload"
    )

    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption="Dein hochgeladenes Bild", use_column_width=True)

        # KI-Analyse (falls Modell geladen)
        if model:
            label, conf = predict(model, labels, img)
            full = is_full(label) and conf >= threshold
            icon = "🔴" if full else "🟢"
            
            # Ergebnis anzeigen
            st.markdown("### 🎯 Analyseergebnis")
            st.markdown(f"""
            <div class="quicktest-card">
                <b>Erkannter Zustand:</b> {icon} {label}<br>
                <b>Konfidenz:</b> {conf*100:.1f}%
            </div>
            """, unsafe_allow_html=True)

            if full:
                st.error("⚠️ **Achtung:** Mülleimer ist voll!")
            else:
                st.success("✅ **Alles okay:** Mülleimer ist nicht voll.")
        else:
            st.warning("Kein Modell geladen – Analyse nicht möglich.")
