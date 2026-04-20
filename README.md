# 🗑️ Trash Monitor

KI-gestützte Echtzeit-Überwachung von Mülleimern mit Streamlit und Teachable Machine.

---

## Was macht diese App?

- Überwacht **mehrere Kameras gleichzeitig**
- Erkennt automatisch, ob ein Mülleimer **voll oder nicht voll** ist
- Zeigt eine **visuelle Warnung + Tonalarm**, wenn ein Eimer überfüllt ist
- Überfüllte Eimer werden **groß und hervorgehoben** angezeigt
- Aktualisiert sich automatisch alle 10–300 Sekunden

---

## Voraussetzungen

- Python 3.9 oder neuer
- Eine Kamera (oder ein Skript, das Bilder in einen Ordner speichert)
- Ein trainiertes [Teachable Machine](https://teachablemachine.withgoogle.com) Modell

---

## Installation

```bash
git clone https://github.com/DEIN_USERNAME/trash-monitor.git
cd trash-monitor
pip install -r requirements.txt
streamlit run app.py
```

---

## Teachable Machine Modell erstellen

1. Gehe zu [teachablemachine.withgoogle.com](https://teachablemachine.withgoogle.com)
2. Wähle **Image Project → Standard Image Model**
3. Erstelle zwei Klassen:
   - `voll` — Fotos von überfüllten Mülleimern
   - `nicht voll` — Fotos von leeren / halbvollen Mülleimern
4. Trainiere das Modell (mind. 50–100 Bilder pro Klasse empfohlen)
5. Klicke auf **Modell exportieren → Tensorflow → Keras**
6. Lade `keras_model.h5` und `labels.txt` herunter

Diese beiden Dateien lädst du dann in der App hoch (Sidebar links).

---

## Kamera einrichten

### Option A: Raspberry Pi / beliebiger PC mit Kamera

Erstelle ein einfaches Skript, das alle 10 Minuten ein Bild speichert:

```python
# raspberry_capture.py
import cv2
import time
import os
from datetime import datetime

CAMERA_ID = 0  # 0 = erste Kamera
SAVE_FOLDER = "/pfad/zur/trash-monitor/camera_images/cam_1"
INTERVAL = 600  # 10 Minuten

os.makedirs(SAVE_FOLDER, exist_ok=True)
cap = cv2.VideoCapture(CAMERA_ID)

while True:
    ret, frame = cap.read()
    if ret:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SAVE_FOLDER, f"capture_{ts}.jpg")
        cv2.imwrite(path, frame)
        print(f"Bild gespeichert: {path}")
    time.sleep(INTERVAL)
```

Starten mit:
```bash
python raspberry_capture.py
```

### Option B: Simulator (zum Testen)

```bash
python camera_simulator.py --camera cam_1 --source /ordner/mit/testbildern --interval 30
```

---

## Mehrere Kameras

Für jede Kamera einen eigenen Unterordner anlegen:

```
camera_images/
├── cam_1/    ← Kamera Eingang
├── cam_2/    ← Kamera Parkplatz
└── cam_3/    ← Kamera Spielplatz
```

Entweder manuell anlegen oder in der App-Sidebar unter **„Neuen Kamera-Ordner anlegen"**.

Für jede Kamera ein separates Capture-Skript mit dem passenden Ordnernamen starten.

---

## Projektstruktur

```
trash-monitor/
├── app.py                  ← Streamlit-App (Hauptdatei)
├── camera_simulator.py     ← Zum Testen ohne echte Kamera
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml         ← Theme-Einstellungen
├── model/                  ← Teachable Machine Modell (nach Upload)
│   ├── keras_model.h5
│   └── labels.txt
└── camera_images/          ← Bilder der Kameras
    ├── cam_1/
    ├── cam_2/
    └── cam_3/
```

---

## Einstellungen in der App

| Einstellung | Beschreibung |
|---|---|
| Schwellenwert | Ab welcher Konfidenz (0.5–1.0) gilt ein Eimer als voll |
| Auto-Refresh | Wie oft die App neue Bilder prüft (10s – 5min) |
| Modell hochladen | keras_model.h5 + labels.txt aus Teachable Machine |

---

## Tipps für bessere Erkennung

- Fotografiere immer **aus der gleichen Perspektive** (Kamera fest montiert)
- Nutze **gute Beleuchtung** — schlechtes Licht verschlechtert die Erkennung
- Trainiere mit Bildern aus **echten Bedingungen** (Tag/Nacht, verschiedene Füllstände)
- Mind. **80 Bilder pro Klasse** für gute Genauigkeit
- Teste dein Modell in Teachable Machine bevor du es hochlädst

---

## Lizenz

MIT License — frei verwendbar, auch für kommerzielle Projekte.
