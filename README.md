# 🗑️ Trash Monitor

KI-gestützte Echtzeit-Überwachung von Mülleimern — gebaut mit Streamlit und Teachable Machine.

---

## Einrichtung

### 1. Teachable Machine Modell
1. Gehe zu [teachablemachine.withgoogle.com](https://teachablemachine.withgoogle.com)
2. **Image Project → Standard Image Model**
3. Zwei Klassen anlegen: `voll` und `nicht voll`
4. Exportieren: **Tensorflow → Keras**
5. `keras_model.h5` und `labels.txt` in den Ordner `model/` legen und auf GitHub pushen

### 2. GitHub Personal Access Token (PAT)
1. GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens**
2. Berechtigungen: **Contents: Read and Write**
3. Token kopieren

### 3. Streamlit Secrets konfigurieren
Auf [share.streamlit.io](https://share.streamlit.io) → Deine App → **Settings → Secrets**:

```toml
GITHUB_TOKEN  = "ghp_DEIN_TOKEN"
GITHUB_REPO   = "dein-username/trash-monitor"
GITHUB_BRANCH = "main"
```

---

## Projektstruktur

```
trash-monitor/
├── app.py
├── requirements.txt
├── cameras.json          ← wird automatisch erstellt
├── .streamlit/
│   └── config.toml
├── model/
│   ├── keras_model.h5    ← von Teachable Machine
│   └── labels.txt        ← von Teachable Machine
└── camera_images/
    └── cam_xyz/
        └── latest.jpg    ← wird automatisch überschrieben
```

---

## Kamera anschließen

Für jede Kamera ein Skript auf dem jeweiligen Gerät (Raspberry Pi, PC, ...) starten:

```python
# capture.py
import cv2, requests, base64, time

GITHUB_TOKEN = "ghp_..."
GITHUB_REPO  = "username/trash-monitor"
CAM_ID       = "cam_1"         # muss mit der ID in der App übereinstimmen
INTERVAL     = 900             # 15 Minuten

def push_image(img_bytes):
    path = f"camera_images/{CAM_ID}/latest.jpg"
    r = requests.get(
        f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}",
        headers={"Authorization": f"token {GITHUB_TOKEN}"}
    )
    sha = r.json().get("sha") if r.status_code == 200 else None
    payload = {
        "message": f"Update {CAM_ID}",
        "content": base64.b64encode(img_bytes).decode(),
        "branch": "main"
    }
    if sha:
        payload["sha"] = sha
    requests.put(
        f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}",
        headers={"Authorization": f"token {GITHUB_TOKEN}"},
        json=payload
    )

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        _, buf = cv2.imencode(".jpg", frame)
        push_image(buf.tobytes())
        print(f"Bild hochgeladen ({CAM_ID})")
    time.sleep(INTERVAL)
```

---

## Features

- 📺 **Monitor-Tab** — Echtzeit-Übersicht aller Kameras, Alarm bei überfülltem Eimer
- 📷 **Kameras verwalten** — Kameras hinzufügen/entfernen, Verbindungsanleitung
- 🧪 **Test-Upload** — manuell ein Bild hochladen und Modell testen
- 🔔 **Ton-Alarm** — Signalton wenn ein Eimer voll erkannt wird
- 💾 **GitHub-Speicherung** — immer nur das letzte Bild pro Kamera, kein Vollaufen

---

## Lizenz
MIT
