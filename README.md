# Trash Monitor - Mülleimer Überwachungssystem

## Setup

### 1. GitHub Token erstellen
- Gehe zu https://github.com/settings/tokens
- Erstelle einen neuen Personal Access Token (Classic)
- Gib ihm die Berechtigung `repo`
- Kopiere den Token

### 2. Secrets konfigurieren

#### Lokal (für Tests):
Erstelle `.streamlit/secrets.toml`:
```toml
GITHUB_TOKEN = "ghp_xxxxxxxxxxxxx"
GITHUB_REPO = "Benexelus/Informatikprojekt-2026"
GITHUB_BRANCH = "main"
```

#### Streamlit Cloud:
Gehe zu App → Settings → Secrets und füge hinzu:
```toml
GITHUB_TOKEN = "ghp_xxxxxxxxxxxxx"
GITHUB_REPO = "Benexelus/Informatikprojekt-2026"
GITHUB_BRANCH = "main"
```

### 3. Keras Modell hochladen
1. Gehe zur App
2. Sidebar → "🧠 Modell hochladen"
3. Wähle deine trainierte `keras_model.h5` Datei
4. Klick auf "💾 Modell speichern"

Die labels.txt wird NICHT überschrieben — sie bleibt im Repo stabil.

## Features
- 📺 **Monitor**: Übersicht aller Kameras in Echtzeit
- 📷 **Kameras verwalten**: Neue Kameras hinzufügen, Bilder hochladen
- 🧪 **Test-Upload**: Modell testen ohne echte Kamera

## Troubleshooting

### "GitHub nicht konfiguriert"
→ Prüfe, dass GITHUB_TOKEN, GITHUB_REPO und GITHUB_BRANCH in secrets.toml gesetzt sind

### "Bad credentials (401)"
→ Token ist ungültig/abgelaufen. Erstelle einen neuen unter https://github.com/settings/tokens

### "model/labels.txt nicht gefunden"
→ Die Datei `model/labels.txt` muss existieren. Sie wird automatisch erstellt, aber du kannst sie auch manuell hochladen.
