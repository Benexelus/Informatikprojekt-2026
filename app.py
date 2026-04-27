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
/* Sidebar navigation buttons */
div[data-testid="stSidebar"] .nav-btn button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.95rem;
    margin-bottom: 4px;
    cursor: pointer;
}
div[data-testid="stSidebar"] .nav-btn-active button {
    background: #ff4b4b22 !important;
    border-left: 3px solid #ff4b4b !important;
    font-weight: 600;
}

/* Cards */
.full-card {
    border: 3px solid #ff4b4b;
    border-radius: 12px;
    padding: 1rem;
    background: #fff5f5;
    box-shadow: 0 0 18px #ff4b4b33;
    margin-bottom: 1rem;
    transition: box-shadow 0.3s;
}
.ok-card {
    border: 1.5px solid #28a745;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.idle-card {
    border: 1.5px solid #ccc;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    background: #fafafa;
}
.alert-banner {
    background: #fff0f0;
    border: 2px solid #ff4b4b;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
}
.status-full  { color: #dc3545; font-weight: 700; font-size: 1rem; }
.status-ok    { color: #28a745; font-weight: 700; font-size: 1rem; }
.status-none  { color: #888;    font-style: italic; font-size: 0.9rem; }
.section-title { font-size: 1.4rem; font-weight: 600; margin-bottom: 1rem; }

/* Connection type badge */
.conn-badge {
    display: inline-block;
    background: #f0f2f6;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.8rem;
    color: #555;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── GitHub helpers ────────────────────────────────────────────────────────────
def _gh_token():  return st.secrets.get("GITHUB_TOKEN", "")
def _gh_repo():   return st.secrets.get("GITHUB_REPO", "")
def _gh_branch(): return st.secrets.get("GITHUB_BRANCH", "main")
def _gh_ok():     return bool(_gh_token() and _gh_repo())

def _gh_headers():
    return {"Authorization": f"token {_gh_token()}",
            "Accept": "application/vnd.github.v3+json"}

def gh_get(path):
    url = f"https://api.github.com/repos/{_gh_repo()}/contents/{path}?ref={_gh_branch()}"
    r = requests.get(url, headers=_gh_headers(), timeout=10)
    if r.status_code == 200:
        d = r.json()
        return base64.b64decode(d["content"]), d["sha"]
    return None, None

def gh_put(path, data: bytes, sha, msg: str):
    url = f"https://api.github.com/repos/{_gh_repo()}/contents/{path}"
    payload = {"message": msg,
               "content": base64.b64encode(data).decode(),
               "branch": _gh_branch()}
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_gh_headers(), json=payload, timeout=15)
    return r.status_code in (200, 201)

def load_cameras() -> dict:
    content, _ = gh_get("cameras.json")
    if content:
        return json.loads(content.decode())
    return {}

def save_cameras(cameras: dict):
    _, sha = gh_get("cameras.json")
    gh_put("cameras.json",
           json.dumps(cameras, indent=2).encode(),
           sha, "Update camera registry")

def save_image(cam_id: str, img_bytes: bytes):
    path = f"camera_images/{cam_id}/latest.jpg"
    _, sha = gh_get(path)
    return gh_put(path, img_bytes, sha, f"Update image {cam_id}")

def load_image(cam_id: str) -> Image.Image | None:
    for ext in ("jpg", "jpeg", "png"):
        content, _ = gh_get(f"camera_images/{cam_id}/latest.{ext}")
        if content:
            return Image.open(BytesIO(content))
    return None

# ── Model ─────────────────────────────────────────────────────────────────────
np.set_printoptions(suppress=True)

def _find_model_path():
    """Sucht Modelldatei, gibt (model_path, labels_path, fehler_str) zurück."""
    if not os.path.exists("model"):
        return None, None, "Ordner `model/` existiert nicht auf dem Server."
    files = os.listdir("model")
    if not files:
        return None, None, "Ordner `model/` ist leer. Bitte `keras_model.h5` und `labels.txt` ins Repo laden."
    model_path = None
    for candidate in ("model/keras_model.h5", "model/keras_Model.h5"):
        if os.path.exists(candidate):
            model_path = candidate
            break
    if model_path is None:
        return None, None, f"Keine `.h5` Datei gefunden. Vorhandene Dateien: `{files}`"
    if not os.path.exists("model/labels.txt"):
        return None, None, f"`labels.txt` fehlt. Vorhandene Dateien: `{files}`"
    return model_path, "model/labels.txt", None

@st.cache_resource
def load_model():
    model_path, labels_path, err = _find_model_path()
    if err:
        return None, None, err
    try:
        import tensorflow as tf
        m = tf.keras.models.load_model(model_path, compile=False)
        with open(labels_path, "r") as f:
            lbl = f.readlines()
        return m, lbl, None
    except Exception as e:
        return None, None, f"Laden fehlgeschlagen: {str(e)}"

def predict(model, class_names, img: Image.Image):
    """Exakt der Teachable Machine Predict-Code."""
    from PIL import ImageOps
    # Array vorbereiten — genau wie Teachable Machine
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Bild auf mindestens 224x224 skalieren, dann mittig croppen
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    # Normalisierung exakt wie Teachable Machine
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    # Vorhersage
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])
    # Label bereinigen: "0 voll" → "voll"
    clean_label = class_name.strip()[2:] if len(class_name.strip()) > 2 else class_name.strip()
    return clean_label, confidence_score, prediction[0]

def is_full(label: str) -> bool:
    return "voll" in label.lower() or "full" in label.lower()

def img_to_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "monitor"
if "cameras" not in st.session_state:
    st.session_state.cameras = load_cameras() if _gh_ok() else {}
if "detail_cam" not in st.session_state:
    st.session_state.detail_cam = None
if "add_cam_open" not in st.session_state:
    st.session_state.add_cam_open = False

model, labels, _model_err = load_model()

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗑️ Trash Monitor")
    st.markdown("---")

    pages = {
        "monitor":  "📺  Monitor",
        "cameras":  "📷  Kameras verwalten",
        "test":     "🧪  Test-Upload",
    }
    for key, label in pages.items():
        active = st.session_state.page == key
        css = "nav-btn nav-btn-active" if active else "nav-btn"
        st.markdown(f'<div class="{css}">', unsafe_allow_html=True)
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Status indicators
    st.markdown("**Status**")
    if _gh_ok():
        st.markdown("🟢 GitHub verbunden")
    else:
        st.markdown("🔴 GitHub nicht konfiguriert")
    if model:
        st.markdown("🟢 KI-Modell geladen")
    else:
        st.markdown("🔴 Kein Modell gefunden")
        if _model_err:
            st.caption(f"ℹ️ {_model_err}")

    st.markdown("---")
    if st.button("🔄 Neu laden", use_container_width=True):
        if _gh_ok():
            st.session_state.cameras = load_cameras()
        st.rerun()

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MONITOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "monitor":
    st.markdown('<div class="section-title">📺 Monitor</div>', unsafe_allow_html=True)

    threshold = st.slider("Schwellenwert (Konfidenz)", 0.5, 1.0, 0.75, 0.05,
                          key="thresh_monitor",
                          help="Ab wann gilt ein Eimer als voll?")

    cameras = st.session_state.cameras
    if not cameras:
        st.info("Noch keine Kameras verbunden. Gehe zu **Kameras verwalten**.")
        st.stop()

    # Analyse
    results = []
    with st.spinner("Bilder werden geladen..."):
        for cam_id, info in cameras.items():
            img = load_image(cam_id) if _gh_ok() else None
            label, conf, full = None, None, False
            if img and model:
                label, conf, _ = predict(model, labels, img)
                full = is_full(label) and conf >= threshold
            results.append(dict(id=cam_id, name=info.get("name", cam_id),
                                connection=info.get("connection", ""),
                                img=img, label=label, conf=conf, full=full))

    full_cams = [r for r in results if r["full"]]

    # Alert banner + sound
    if full_cams:
        names = " &nbsp;|&nbsp; ".join(["📍 " + r["name"] for r in full_cams])
        st.markdown(
            f'<div class="alert-banner"><h3>🚨 {len(full_cams)} Mülleimer überfüllt!</h3>'
            f'<p>{names}</p></div>',
            unsafe_allow_html=True)
        st.components.v1.html("""<script>
        try {
            const ctx = new AudioContext();
            const beep = (f,d,v,t) => {
                const o=ctx.createOscillator(), g=ctx.createGain();
                o.connect(g); g.connect(ctx.destination);
                o.frequency.value=f;
                g.gain.setValueAtTime(v, ctx.currentTime+t);
                g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime+t+d);
                o.start(ctx.currentTime+t); o.stop(ctx.currentTime+t+d+0.05);
            };
            beep(880,0.25,0.4,0); beep(880,0.25,0.4,0.35); beep(1200,0.4,0.5,0.7);
        } catch(e) {}
        </script>""", height=0)

    # Überfüllte Eimer — prominent
    if full_cams:
        st.markdown("### 🔴 Überfüllte Eimer")
        cols = st.columns(min(len(full_cams), 3))
        for i, r in enumerate(full_cams):
            with cols[i % 3]:
                st.markdown('<div class="full-card">', unsafe_allow_html=True)
                st.markdown(f"**🗑️ {r['name']}**")
                st.image(r["img"], use_column_width=True)
                st.markdown(f'<div class="status-full">🔴 {r["label"]} — {r["conf"]*100:.1f}%</div>',
                            unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

    # Alle Kameras
    st.markdown("### 📷 Alle Kameras")
    cols = st.columns(min(len(results), 3))
    for i, r in enumerate(results):
        with cols[i % 3]:
            card = "full-card" if r["full"] else ("ok-card" if r["img"] else "idle-card")
            st.markdown(f'<div class="{card}">', unsafe_allow_html=True)
            st.markdown(f"**{r['name']}**")
            st.markdown(f'<span class="conn-badge">{r["connection"]}</span>', unsafe_allow_html=True)

            if r["img"]:
                st.image(r["img"], use_column_width=True)
                if r["label"]:
                    icon = "🔴" if r["full"] else "🟢"
                    cls  = "status-full" if r["full"] else "status-ok"
                    st.markdown(f'<div class="{cls}">{icon} {r["label"]} ({r["conf"]*100:.1f}%)</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-none">⚠️ Kein Modell: {_model_err or "unbekannter Fehler"}</div>',
                                unsafe_allow_html=True)

                if st.button("🔍 Detail", key=f"detail_{r['id']}"):
                    st.session_state.detail_cam = r["id"]
                    st.rerun()
            else:
                st.markdown('<div class="status-none">⏳ Noch kein Bild empfangen</div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Detail modal
    if st.session_state.detail_cam:
        cam_id = st.session_state.detail_cam
        info   = cameras.get(cam_id, {})
        st.markdown("---")
        st.markdown(f"### 🔍 Detail: {info.get('name', cam_id)}")
        img = load_image(cam_id)
        if img:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(img, use_column_width=True)
            with c2:
                st.markdown(f"**Kamera:** {info.get('name', cam_id)}")
                st.markdown(f"**Verbindung:** {info.get('connection', '—')}")
                st.markdown(f"**Hinzugefügt:** {info.get('added', '—')[:10]}")
                if model:
                    lbl, conf, _ = predict(model, labels, img)
                    full = is_full(lbl) and conf >= threshold
                    st.metric("Ergebnis", f"{'🔴' if full else '🟢'} {lbl}",
                              f"{conf*100:.1f}% Konfidenz")
                    if full:
                        st.error("⚠️ Überfüllt — bitte leeren!")
                    else:
                        st.success("✅ Nicht überfüllt")
        else:
            st.warning("Kein Bild verfügbar.")
        if st.button("✖ Schließen"):
            st.session_state.detail_cam = None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: KAMERAS VERWALTEN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "cameras":
    st.markdown('<div class="section-title">📷 Kameras verwalten</div>', unsafe_allow_html=True)

    threshold = st.slider("Schwellenwert (Konfidenz)", 0.5, 1.0, 0.75, 0.05,
                          key="thresh_cameras")

    # ── Neue Kamera ───────────────────────────────────────────────────────────
    with st.expander("➕ Neue Kamera hinzufügen", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            new_name = st.text_input("Name der Kamera", placeholder="z.B. Eingang Rathaus")
            new_id   = st.text_input("Kamera-ID (keine Leerzeichen, keine Umlaute)",
                                     placeholder="z.B. cam_rathaus")
        with c2:
            conn_type = st.selectbox("Verbindungstyp", [
                "Raspberry Pi (HTTP)",
                "PC / Laptop (lokal)",
                "ESP32-CAM (LTE / WLAN)",
                "Smartphone (Termux)",
                "Bluetooth Relay",
                "Manueller Upload",
            ])
            conn_note = st.text_input("Zusatzinfo (optional)",
                                      placeholder="z.B. IP-Adresse, Standort")

        if st.button("✅ Kamera hinzufügen", type="primary"):
            if not new_name or not new_id:
                st.error("Bitte Name und ID eingeben.")
            elif " " in new_id:
                st.error("Die Kamera-ID darf keine Leerzeichen enthalten.")
            elif new_id in st.session_state.cameras:
                st.error(f"ID '{new_id}' existiert bereits.")
            else:
                st.session_state.cameras[new_id] = {
                    "name": new_name,
                    "connection": f"{conn_type}" + (f" — {conn_note}" if conn_note else ""),
                    "added": datetime.now().isoformat(),
                }
                if _gh_ok():
                    with st.spinner("Wird auf GitHub gespeichert..."):
                        save_cameras(st.session_state.cameras)
                st.success(f"✅ Kamera **{new_name}** hinzugefügt!")
                st.rerun()

    # ── Bild manuell hochladen ────────────────────────────────────────────────
    if st.session_state.cameras:
        st.markdown("---")
        st.markdown("### 📤 Bild für Kamera hochladen")
        st.caption("Manuell ein Bild einer bestehenden Kamera hochladen — z.B. als einmaliger Test.")

        cam_opts = {v["name"]: k for k, v in st.session_state.cameras.items()}
        sel_name = st.selectbox("Kamera wählen", list(cam_opts.keys()), key="upload_cam_select")
        sel_id   = cam_opts[sel_name]
        up_file  = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"],
                                    key="cam_upload")

        if up_file:
            img = Image.open(up_file)
            st.image(img, width=300)

            if model:
                lbl, conf, _ = predict(model, labels, img)
                full = is_full(lbl) and conf >= threshold
                st.metric("KI-Vorschau", f"{'🔴' if full else '🟢'} {lbl}",
                          f"{conf*100:.1f}%")

            if st.button("📤 Hochladen & speichern", type="primary"):
                if _gh_ok():
                    with st.spinner("Wird auf GitHub gespeichert..."):
                        ok = save_image(sel_id, img_to_bytes(img))
                    if ok:
                        st.success("✅ Bild gespeichert!")
                    else:
                        st.error("Fehler beim Speichern auf GitHub.")
                else:
                    st.warning("GitHub nicht konfiguriert — Speichern nicht möglich.")

    # ── Verbindungsanleitung ──────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📡 Verbindungsanleitung — wie schicke ich Bilder an die App?"):
        st.markdown("""
### Wie funktioniert es?
Jede Kamera läuft als kleines Python-Skript auf ihrem Gerät. Das Skript macht alle 15–30 Minuten ein Foto und lädt es direkt auf GitHub hoch — die App liest es dann von dort.

---

### Option 1 — Raspberry Pi oder PC (empfohlen)
```python
# capture.py
import cv2, requests, base64, time

GITHUB_TOKEN = "ghp_DEIN_TOKEN"
GITHUB_REPO  = "username/trash-monitor"
CAM_ID       = "cam_1"      # muss mit der ID in der App übereinstimmen
INTERVAL     = 900          # Sekunden (900 = 15 Minuten)

def push(img_bytes):
    path = f"camera_images/{CAM_ID}/latest.jpg"
    r = requests.get(f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}",
                     headers={"Authorization": f"token {GITHUB_TOKEN}"})
    sha = r.json().get("sha") if r.status_code == 200 else None
    payload = {"message": f"Update {CAM_ID}",
               "content": base64.b64encode(img_bytes).decode(),
               "branch": "main"}
    if sha: payload["sha"] = sha
    requests.put(f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}",
                 headers={"Authorization": f"token {GITHUB_TOKEN}"}, json=payload)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        _, buf = cv2.imencode(".jpg", frame)
        push(buf.tobytes())
        print("Bild hochgeladen")
    time.sleep(INTERVAL)
```

---

### Option 2 — ESP32-CAM (LTE / WLAN)
ESP32-CAM sendet per HTTP POST an einen kleinen Relay-Server, der dann auf GitHub pusht.

---

### Option 3 — Smartphone (Termux / Android)
Termux installieren → Python installieren → obiges Skript mit der Handykamera nutzen.

```bash
# In Termux:
pkg install python
pip install requests opencv-python
python capture.py
```

---

### Option 4 — Bluetooth Relay
Kamera sendet per Bluetooth an einen Raspberry Pi in der Nähe, der dann per WLAN auf GitHub pusht.

---

### Option 5 — Manueller Upload
Kein Skript nötig — Bild direkt über die App hochladen (oben auf dieser Seite).
        """)

    # ── Bestehende Kameras ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Verbundene Kameras")
    cameras = st.session_state.cameras
    if not cameras:
        st.info("Noch keine Kameras hinzugefügt.")
    else:
        for cam_id, info in list(cameras.items()):
            c1, c2, c3, c4 = st.columns([3, 3, 1, 1])
            with c1:
                st.markdown(f"**{info['name']}**  \n`{cam_id}`")
            with c2:
                st.markdown(f'<span class="conn-badge">{info.get("connection","")}</span>',
                            unsafe_allow_html=True)
            with c3:
                # Show last image
                if st.button("🖼 Bild", key=f"show_{cam_id}"):
                    st.session_state.detail_cam = cam_id
                    st.session_state.page = "monitor"
                    st.rerun()
            with c4:
                if st.button("🗑 Löschen", key=f"del_{cam_id}"):
                    del st.session_state.cameras[cam_id]
                    if _gh_ok():
                        save_cameras(st.session_state.cameras)
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TEST UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "test":
    st.markdown('<div class="section-title">🧪 Test-Upload</div>', unsafe_allow_html=True)
    st.markdown("Lade ein Bild hoch um zu prüfen, ob das Modell es korrekt einschätzt — ohne dass eine echte Kamera verbunden sein muss.")

    threshold = st.slider("Schwellenwert (Konfidenz)", 0.5, 1.0, 0.75, 0.05,
                          key="thresh_test")

    st.markdown("---")

    uploaded = st.file_uploader("📁 Bild hochladen (JPG / PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("**Hochgeladenes Bild**")
            st.image(img, use_column_width=True)

        with c2:
            st.markdown("**KI-Auswertung**")
            if model:
                with st.spinner("Analysiere..."):
                    lbl, conf, all_preds = predict(model, labels, img)
                    full = is_full(lbl) and conf >= threshold

                icon = "🔴" if full else "🟢"
                st.metric("Ergebnis", f"{icon} {lbl}", f"{conf*100:.1f}% Konfidenz")

                # Confidence bar
                st.progress(conf)

                if full:
                    st.error("⚠️ **Überfüllt** — Dieser Mülleimer sollte geleert werden.")
                else:
                    st.success("✅ **Nicht überfüllt** — Alles in Ordnung.")

                # All class probabilities — direkt aus predict(), kein zweiter model.predict() nötig
                st.markdown("**Alle Klassen:**")
                for i, lbl_name in enumerate(labels):
                    clean = lbl_name.strip()[2:] if len(lbl_name.strip()) > 2 else lbl_name.strip()
                    st.progress(float(all_preds[i]), text=f"{clean}: {all_preds[i]*100:.1f}%")

            else:
                st.warning(f"⚠️ Kein KI-Modell gefunden.  \n{_model_err or ''}")

            st.markdown("---")

            # Optionally save to a camera
            st.markdown("**Optional: Als Kamera-Bild speichern**")
            cameras = st.session_state.cameras
            if cameras and _gh_ok():
                cam_opts = {v["name"]: k for k, v in cameras.items()}
                sel = st.selectbox("Kamera wählen", list(cam_opts.keys()), key="test_cam_sel")
                if st.button("💾 Als letztes Bild dieser Kamera speichern"):
                    with st.spinner("Speichere..."):
                        ok = save_image(cam_opts[sel], img_to_bytes(img))
                    if ok:
                        st.success(f"✅ Gespeichert für **{sel}**")
                    else:
                        st.error("Fehler beim Speichern.")
            elif not cameras:
                st.caption("Erst unter **Kameras verwalten** eine Kamera anlegen.")
            else:
                st.caption("GitHub nicht konfiguriert — Speichern nicht möglich.")
    else:
        # Tips when nothing uploaded yet
        st.markdown("### 💡 Tipps für gute Testergebnisse")
        st.markdown("""
- Fotografiere den Mülleimer **aus der gleichen Perspektive** wie die echte Kamera
- Achte auf **gute Beleuchtung** — schlechtes Licht verschlechtert die Erkennung deutlich
- Teste sowohl **volle** als auch **leere** Eimer
- Der Schwellenwert oben bestimmt, ab welcher Sicherheit ein Eimer als "voll" gilt
- Wenn das Modell oft falsch liegt, trainiere es mit mehr Bildern nach
        """)
