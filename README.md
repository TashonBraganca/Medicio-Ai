# 🏥 Medo-AI — Offline Medical Assistant

> A privacy-first, local AI assistant for symptom triage, clinical image review, and medical document interpretation. Runs with Ollama models and a Streamlit interface.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Live demo (if deployed)

Try a hosted demo (when available): https://your-deployed-url.streamlit.app

![Medo-AI Demo](f650c04eef1b4bf428e9.png)

---

## What Medo-AI does

Medo-AI is built to give rapid, structured medical guidance locally on your machine. It combines a Streamlit front end with Ollama-powered local models to provide:

- Conversational symptom assessment and first-aid guidance.
- OCR extraction and interpretation of lab reports, prescriptions, and PDFs.
- Clinical image triage (wounds, rashes, burns) with urgency indicators.
- Local-only inference to keep patient data private (no default external API calls).

The repository contains the Streamlit app (app.py), model/config settings (config.py), and modular services under services/.

---

## Highlights

- Conversational assistant with structured responses: likely causes, immediate steps, and escalation criteria.
- Document intelligence: OCR, abnormal-value highlighting, and plain-language summaries.
- Image triage: urgency scoring, care recommendations, and vision-model support for richer output.
- Safety guard: automatic red-flag detection and escalation suggestions.
- Privacy-first: default on-device processing using Ollama.

---

## Quick start — Local

Prerequisites

- Python 3.8+
- Ollama installed and on PATH
- 8GB RAM minimum (16GB recommended for vision models)
- Optional: Tesseract OCR for improved document parsing

1) Install Ollama

Windows
```
# via winget
winget install Ollama.Ollama
# or grab installer from https://ollama.com/download
```

macOS
```
brew install ollama
# or download from https://ollama.com/download
```

Linux
```
curl -fsSL https://ollama.com/install.sh | sh
```

Restart your terminal after installation so `ollama` is available in PATH.

2) Pull recommended models (adjust for your hardware)

```
ollama pull gemma2:2b     # Recommended primary model (fast & compact)
ollama pull qwen2:1.5b    # Lightweight fallback
ollama pull llava:7b      # Optional: vision-capable model for image analysis
```

3) Clone repo & install Python deps

```
git clone https://github.com/Pratham-r05/Medo-AI.git
cd Medo-AI
pip install -r requirements.txt
```

4) (Optional) Install Tesseract OCR

Windows: download UB Mannheim build and add to PATH.
macOS:
```
brew install tesseract
```
Linux:
```
sudo apt-get install tesseract-ocr
```

5) Run the application

```
streamlit run app.py
```

Open http://localhost:8501 (Streamlit may choose another free port if 8501 is busy).

---

## Configuration & customization

- Edit config.py to change default model names, thresholds, or prompt templates.
- The app will try the primary model and fall back automatically if not available.
- Enable vision features only if you have a vision-capable model and sufficient memory.

---

## Project layout

- app.py — Streamlit UI and flow control
- config.py — model names, prompt text, and thresholds
- requirements.txt — Python dependencies
- services/
  - chat_service.py — LLM chat handling and formatting
  - document_service.py — OCR and PDF parsing utilities
  - vision_service.py — image preprocessing and vision model integration
  - safety_guard.py — red-flag detection and scope checks
- f650c04eef1b4bf428e9.png — demo screenshot used in README

---

## Troubleshooting

- "ollama: command not found": ensure Ollama is installed and PATH updated; restart terminal.
- "Model not listed": run `ollama list` and re-pull models with `ollama pull <model>`.
- Streamlit port conflict: run with `streamlit run app.py --server.port <PORT>`.
- Poor OCR results: confirm Tesseract is installed and language packs are present.

---

## Safety, privacy & limitations

- Medo-AI is an assistive tool and not a replacement for clinical judgement.
- Always escalate suspected life-threatening conditions to emergency services.
- The default configuration prefers local processing; if you add integrations that transmit data off-device, review privacy implications carefully.

Important: This software is for informational use only and is not a medical device. Consult licensed healthcare professionals for diagnosis and treatment.

---

## Contributing

Contributions are welcome. Ways to help:

- Improve medical prompts and accuracy
- Add localization and language support
- Enhance vision pipelines and labeling
- Add tests, CI, and packaging

Open issues or PRs on GitHub; refer to CONTRIBUTING.md if present.

---

## Dependencies

See requirements.txt for pinned versions. Typical packages include streamlit, opencv-python, Pillow, pytesseract, PyPDF2, pandas, and numpy.

---

## Support

- Report bugs or request features via GitHub Issues.
- Streamlit community forum for deployment questions.
- Check IMPROVEMENTS.md for ideas and roadmap notes.

---

## License

MIT — see the LICENSE file for details.

---

Built with a focus on local-first, privacy-preserving medical AI — powered by Streamlit & Ollama.