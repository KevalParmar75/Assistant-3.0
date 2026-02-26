# 🤖 Optimus: Multilingual AI Voice Assistant

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/AI-Qwen%2072B-orange.svg" alt="AI Model">
  <img src="https://img.shields.io/badge/UI-CustomTkinter-brightgreen.svg" alt="UI Framework">
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
</div>

<br>

> **Optimus** is an advanced, multilingual, desktop-based AI voice assistant. Powered by the massive **Qwen2.5-72B-Instruct** model via Hugging Face, it features a slick animated HUD, native application control, and seamless YouTube integration.

---

## ✨ Features

- 🌍 **Multilingual Support:** Converses fluently in English (ENG), Hindi (HIN), and Gujarati (GUJ).
- 🧠 **Heavyweight AI Brain:** Uses `Qwen2.5-72B-Instruct` for context-aware, highly intelligent responses.
- 🎨 **Futuristic HUD UI:** Built with `CustomTkinter` and `Tkinter` canvas, featuring real-time state animations (Standby, Listening, Processing, Speaking).
- 🎙️ **Voice Activation:** Press `Left Shift` to trigger listening, or use the wake word "Optimus".
- 🚀 **App Launcher:** Opens local desktop applications natively using natural language.
- 🎵 **Media Controller:** Plays requested songs or searches your liked videos on YouTube.
- 💾 **Persistent Memory:** Remembers the context of your conversation across sessions.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:
* **Python 3.8+**
* **Microphone** connected and configured.
* **Hugging Face API Token** (Free to get from [Hugging Face](https://huggingface.co/settings/tokens)).
* **Google Cloud Console Project** (For YouTube Data API v3 credentials).

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/optimus-assistant.git](https://github.com/yourusername/optimus-assistant.git)
cd optimus-assistant

pip install -r requirements.txt

python optimus.py
```

```bash
Configure Environment Variables
Create a file named .env in the root directory of the project and add your Hugging Face API key so Optimus can access the Qwen model:

Code snippet
HF_TOKEN=your_hugging_face_token_here
Setup YouTube OAuth (For Liked Videos)
To allow Optimus to search and play from your YouTube "Liked" playlist:

Go to the Google Cloud Console.

Create a new project and enable the YouTube Data API v3.

Create OAuth 2.0 Client ID credentials (select "Desktop app").

Download the JSON file, rename it to client_secret.json, and place it in the root directory of this project.
```
🎮 Usage Guide
Once you run python optimus.py, the Optimus HUD will launch. Here is how to control your new assistant:

Move the HUD: Click and drag the circular UI anywhere on your screen.

Switch Languages: Click the ENG, HIN, or GUJ toggles. The HUD colors will change to reflect the active language.

Trigger Listening: * Press the Left Shift key on your keyboard.

Or just say the wake word: "Optimus..."

Example Commands:

🖥️ "Optimus, open Calculator" * 🎵 "Play Blinding Lights on YouTube"

🔍 "Search my liked videos for Python tutorials"

🧠 "What is the distance to the moon?"

⚠️ Troubleshooting & Common Fixes
"ModuleNotFoundError" for PyAudio: If speech recognition fails to build, you likely need PyAudio.

Windows: pip install pyaudio

Linux: sudo apt install portaudio19-dev python3-pyaudio before pip installing.

Mac: brew install portaudio then pip install pyaudio.

YouTube Auth Hangs: On the very first run, a browser window will open asking you to log into your Google Account. Once you approve it, a token.pickle file is generated so you don't have to log in again.

Audio Playing Too Fast/Slow: If pygame throws errors, ensure your system's default audio output is properly configured and not exclusively locked by another app.

🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

<div align="center">
<i>Built with ❤️ by Keval Parmar for the open-source AI community.</i>
</div>
