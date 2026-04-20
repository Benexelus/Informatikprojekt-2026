"""
camera_simulator.py
───────────────────
Simuliert eine Kamera, die alle 10 Minuten ein Bild in einen Ordner speichert.
Zum Testen: Kopiert Bilder aus einem Quellordner in den Zielordner.

Verwendung:
  python camera_simulator.py --camera cam_1 --source /pfad/zu/testbildern --interval 600
"""

import argparse
import os
import shutil
import time
import glob
from datetime import datetime


def simulate_camera(camera_name: str, source_folder: str, interval: int):
    target = os.path.join("camera_images", camera_name)
    os.makedirs(target, exist_ok=True)

    images = sorted(glob.glob(os.path.join(source_folder, "*.jpg")) +
                    glob.glob(os.path.join(source_folder, "*.jpeg")) +
                    glob.glob(os.path.join(source_folder, "*.png")))

    if not images:
        print(f"Keine Bilder in {source_folder} gefunden.")
        return

    print(f"Starte Simulation für {camera_name} | Intervall: {interval}s | Bilder: {len(images)}")
    idx = 0
    while True:
        img = images[idx % len(images)]
        ext = os.path.splitext(img)[1]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(target, f"capture_{ts}{ext}")
        shutil.copy2(img, dest)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {camera_name}: Bild gespeichert → {dest}")
        idx += 1
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kamera-Simulator für Trash Monitor")
    parser.add_argument("--camera", default="cam_1", help="Kamera-Name (= Ordnername)")
    parser.add_argument("--source", required=True, help="Ordner mit Testbildern")
    parser.add_argument("--interval", type=int, default=600, help="Intervall in Sekunden (Standard: 600 = 10 min)")
    args = parser.parse_args()

    simulate_camera(args.camera, args.source, args.interval)
