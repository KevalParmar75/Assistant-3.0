# 🤖 Optimus: Multilingual AI Voice Assistant

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Multimodel-purple.svg)]()
[![UI Framework](https://img.shields.io/badge/UI-CustomTkinter-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange.svg)]()

> **Optimus** is an advanced, multilingual, desktop-based AI voice assistant. Originally powered by **Qwen2.5-72B-Instruct**, Optimus is currently undergoing a **major architectural overhaul** toward a flexible multimodel brain, screen awareness, browser automation, and deep calendar/email integration — while keeping the same futuristic HUD you know.

---

## 🚧 Development Status

> **⚠️ Active Refactor in Progress**
> Optimus is in a **polishing phase** following a major multimodel architecture shift. Core features are working; new capabilities listed below are partially implemented or under active development.

### ✅ Stable & Working
- Multilingual voice interaction (ENG / HIN / GUJ)
- Futuristic animated HUD with state-aware visuals
- Wake word + hotkey activation
- App launcher via natural language
- YouTube search & liked video playback
- Persistent conversation memory

### 🔄 In Progress
- **Multimodel AI Architecture** — Qwen remains the core, with support for additional models being integrated for task-specific routing
- **Screen Awareness** — Optimus can observe and reason about what's on your screen in real time
- **Full Browser Automation** — End-to-end control of browser actions via natural language
- **Calendar & Email Integration** — Reading, drafting, and managing calendar events and emails
- **RAG + PageIndex Memory** — Hybrid retrieval combining normal RAG with a page-indexed document store for richer, context-aware long-term memory

---

## ✨ Features

- 🌍 **Multilingual Support:** Converses fluently in English (ENG), Hindi (HIN), and Gujarati (GUJ)
- 🧠 **Multimodel AI Brain:** Flexible model routing — Qwen2.5-72B-Instruct at the core, with support for additional models
- 🎨 **Futuristic HUD UI:** Built with `CustomTkinter` and `Tkinter` canvas, featuring real-time state animations (Standby, Listening, Processing, Speaking)
- 🎙️ **Voice Activation:** Press `Left Shift` to trigger listening, or use the wake word **"Optimus"**
- 🚀 **App Launcher:** Opens local desktop applications natively using natural language
- 🎵 **Media Controller:** Plays requested songs or searches your liked videos on YouTube
- 💾 **Hybrid Memory System:** Combines persistent conversation context with RAG + PageIndex retrieval for smarter, longer-term awareness
- 🖥️ **Screen Awareness** *(in development)*: Optimus observes and understands your screen in real time
- 🌐 **Browser Automation** *(in development)*: Full control of browser tasks via voice or text commands
- 📅 **Calendar & Email** *(in development)*: Reads and manages your schedule and inbox

---

## 🛠️ Prerequisites

- **Python 3.8+**
- **Microphone** connected and configured
- **Hugging Face API Token** — [Get one here](https://huggingface.co/settings/tokens)
- **Google Cloud Console Project** — For YouTube Data API v3 credentials

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/KevalParmar75/Assistant-3.0.git
cd Assistant-3.0
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
HF_TOKEN=your_hugging_face_token_here
```

### 4. Setup YouTube OAuth (For Liked Videos)

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project and enable the **YouTube Data API v3**
3. Create **OAuth 2.0 Client ID** credentials (select "Desktop app")
4. Download the JSON file, rename it to `client_secret.json`, and place it in the project root

### 5. Run Optimus

```bash
python optimus.py
```

---

## 🎮 Usage Guide

Once launched, the Optimus HUD will appear. Here's how to interact with it:

| Action | How |
|--------|-----|
| Move the HUD | Click and drag anywhere on screen |
| Switch language | Click **ENG**, **HIN**, or **GUJ** toggles |
| Trigger listening | Press **Left Shift** or say **"Optimus"** |

### Example Commands

- 🖥️ `"Optimus, open Calculator"`
- 🎵 `"Play Blinding Lights on YouTube"`
- 🔍 `"Search my liked videos for Python tutorials"`
- 🧠 `"What is the distance to the moon?"`
- 🌐 `"Open Gmail and check my latest emails"` *(coming soon)*
- 📅 `"What's on my calendar tomorrow?"` *(coming soon)*

---

## ⚠️ Troubleshooting

**`ModuleNotFoundError` for PyAudio:**
```bash
# Windows
pip install pyaudio

# Linux
sudo apt install portaudio19-dev python3-pyaudio && pip install pyaudio

# Mac
brew install portaudio && pip install pyaudio
```

**YouTube Auth Hangs on First Run:**  
A browser window will open asking you to sign into Google. Once approved, a `token.pickle` file is saved so you won't need to log in again.

**Audio Playing Too Fast/Slow:**  
Ensure your system's default audio output is properly configured and not exclusively locked by another application.

---

## 🤝 Contributing

Contributions are welcome and greatly appreciated!

1. Fork the Project
2. Create your Feature Branch: `git checkout -b feature/AmazingFeature`
3. Commit your Changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the Branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 🗺️ Roadmap

- [x] Multilingual voice assistant with HUD
- [x] App launcher & YouTube integration
- [x] Persistent conversation memory
- [ ] Multimodel AI routing architecture
- [ ] RAG + PageIndex hybrid memory
- [ ] Screen awareness
- [ ] Full browser automation
- [ ] Calendar & email integration

---

*Built with ❤️ by Keval Parmar for the open-source AI community.*
