import asyncio
import threading
import tkinter as tk
import customtkinter as ctk
import speech_recognition as sr
import pywhatkit
import edge_tts
import pygame
import os
import pickle
import re
import time
import webbrowser
from AppOpener import open as open_app
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Google Auth
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Load environment variables from .env file
load_dotenv()

# ================= CONFIGURATION =================
REPO_ID = "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

VOICES = {
    "en": "en-GB-RyanNeural",
    "hi": "hi-IN-MadhurNeural",
    "gu": "gu-IN-NiranjanNeural"
}


# ================= YOUTUBE CLIENT =================
class YouTubeClient:
    def __init__(self):
        self.creds = None
        self.service = None
        self.authenticate()

    def authenticate(self):
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token: self.creds = pickle.load(token)
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                if not os.path.exists('client_secret.json'): return
                flow = InstalledAppFlow.from_client_secrets_file('client_secret.json',
                                                                 ['https://www.googleapis.com/auth/youtube.readonly'])
                self.creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(self.creds, token)
        if self.creds: self.service = build('youtube', 'v3', credentials=self.creds)

    def search_my_likes(self, query):
        if not self.service: return None
        try:
            request = self.service.playlistItems().list(part="snippet", playlistId="LL", maxResults=50)
            response = request.execute()
            for item in response.get('items', []):
                title = item['snippet']['title'].lower()
                if query.lower() in title:
                    vid_id = item['snippet']['resourceId']['videoId']
                    return f"https://www.youtube.com/watch?v={vid_id}"
        except:
            pass
        return None


# ================= THE BRAIN (QWEN 72B) =================
class OptimusBrain:
    def __init__(self):
        self.yt_client = YouTubeClient()
        self.memory_file = "optimus_memory.pkl"
        self.client = InferenceClient(token=HF_TOKEN)

        self.chat_history = []
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    self.chat_history = pickle.load(f)
            except:
                pass

    def decide_action(self, command, lang_mode="en"):
        try:
            lang_name = "English"
            if lang_mode == "hi":
                lang_name = "Hindi"
            elif lang_mode == "gu":
                lang_name = "Gujarati"

            system_msg = f"""You are Optimus. 
            CURRENT MODE: {lang_name} Language.
            INSTRUCTIONS:
            1. Reply in {lang_name}.
            2. To OPEN app: Output [[OPEN: appname]].
            3. To PLAY song: Output [[PLAY: songname]].
            4. To search LIKED videos: Output [[LIKED: query]].
            5. Keep answers short.
            """

            messages = [{"role": "system", "content": system_msg}]
            for msg in self.chat_history[-6:]: messages.append(msg)
            messages.append({"role": "user", "content": command})

            response = self.client.chat_completion(
                model=REPO_ID,
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()

            self.chat_history.append({"role": "user", "content": command})
            clean_resp = re.sub(r"\[\[.*?\]\]", "", response_text).strip()
            self.chat_history.append({"role": "assistant", "content": clean_resp if clean_resp else "Done."})
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.chat_history, f)

            # --- TOOL EXECUTION ---
            match_open = re.search(r"\[\[OPEN:\s*(.*?)\]\]", response_text)
            if match_open:
                app_name = match_open.group(1).lower()
                # FIX: Handle YouTube specially since it's a website
                if "youtube" in app_name:
                    webbrowser.open("https://www.youtube.com")
                else:
                    try:
                        open_app(app_name, match_closest=True, output=True)
                    except:
                        pass
                response_text = response_text.replace(match_open.group(0), "")

            match_play = re.search(r"\[\[PLAY:\s*(.*?)\]\]", response_text)
            if match_play:
                pywhatkit.playonyt(match_play.group(1))
                response_text = response_text.replace(match_play.group(0), "")

            match_liked = re.search(r"\[\[LIKED:\s*(.*?)\]\]", response_text)
            if match_liked:
                q = match_liked.group(1)
                url = self.yt_client.search_my_likes(q)
                if url:
                    webbrowser.open(url)
                else:
                    if "history" in q.lower() or "mix" in q.lower():
                        pywhatkit.playonyt("My Supermix")
                    else:
                        pywhatkit.playonyt(q)
                response_text = response_text.replace(match_liked.group(0), "")

            return response_text.strip()

        except Exception as e:
            print(f"Error: {e}")
            return "System Error."


# ================= THE HUD UI =================
ctk.set_appearance_mode("Dark")


class OptimusUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.overrideredirect(True)
        self.geometry("400x500")
        self.config(background='#000001')
        self.attributes('-transparentcolor', '#000001')
        self.attributes('-topmost', True)
        self.bind("<Button-1>", self.start_move)
        self.bind("<B1-Motion>", self.do_move)

        self.brain = OptimusBrain()
        self.recognizer = sr.Recognizer()
        self.is_listening = False
        self.current_lang = "en"

        self.rotation_angle = 0
        self.speaking_pulse = 0
        self.pulse_size = 0
        self.pulse_growing = True

        self.attributes("-alpha", 0.0)
        self.fade_in()

        self.canvas = tk.Canvas(self, width=400, height=400, bg='#000001', highlightthickness=0)
        self.canvas.pack()

        self.lang_frame = ctk.CTkFrame(self, fg_color="#111", corner_radius=20)
        self.lang_frame.pack(pady=10)
        self.btn_en = self.create_lang_btn("ENG", "en")
        self.btn_hi = self.create_lang_btn("HIN", "hi")
        self.btn_gu = self.create_lang_btn("GUJ", "gu")
        self.update_lang_buttons()

        self.status_text = "STANDBY"
        self.animate_hud()
        self.bind("<Shift_L>", self.toggle_listening_event)

    def create_lang_btn(self, text, lang_code):
        return ctk.CTkButton(self.lang_frame, text=text, width=60,
                             command=lambda: self.set_language(lang_code),
                             fg_color="#333", hover_color="#555")

    def set_language(self, lang_code):
        self.current_lang = lang_code
        self.update_lang_buttons()

    def update_lang_buttons(self):
        self.btn_en.pack(side="left", padx=5)
        self.btn_hi.pack(side="left", padx=5)
        self.btn_gu.pack(side="left", padx=5)
        self.btn_en.configure(fg_color="#00eaff" if self.current_lang == "en" else "#333",
                              text_color="black" if self.current_lang == "en" else "white")
        self.btn_hi.configure(fg_color="#ff9900" if self.current_lang == "hi" else "#333",
                              text_color="black" if self.current_lang == "hi" else "white")
        self.btn_gu.configure(fg_color="#00ff9c" if self.current_lang == "gu" else "#333",
                              text_color="black" if self.current_lang == "gu" else "white")

    def fade_in(self):
        alpha = self.attributes("-alpha")
        if alpha < 1.0:
            self.attributes("-alpha", alpha + 0.05)
            self.after(30, self.fade_in)

    def start_move(self, event):
        self.x, self.y = event.x, event.y

    def do_move(self, event):
        self.geometry(f"+{self.winfo_x() + event.x - self.x}+{self.winfo_y() + event.y - self.y}")

    def draw_hud(self):
        self.canvas.delete("all")
        cx, cy = 200, 200

        if self.status_text in ["LISTENING...", "AWAITING..."]:
            c, g = "#ff2d2d", "#550000"
        elif self.status_text == "PROCESSING...":
            c, g = "#ffd500", "#554400"
        elif self.status_text == "SPEAKING...":
            c, g = "#00ff9c", "#005533"
        else:
            if self.current_lang == "hi":
                c, g = "#ff9900", "#553300"
            elif self.current_lang == "gu":
                c, g = "#00ff9c", "#005533"
            else:
                c, g = "#00eaff", "#003a44"

        self.canvas.create_arc(cx - 150, cy - 150, cx + 150, cy + 150, start=self.rotation_angle, extent=100, outline=c,
                               width=3, style="arc")
        self.canvas.create_arc(cx - 150, cy - 150, cx + 150, cy + 150, start=self.rotation_angle + 180, extent=100,
                               outline=c, width=3, style="arc")
        r = 70 + self.pulse_size + self.speaking_pulse
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=c, width=3)
        self.canvas.create_oval(cx - (r - 18), cy - (r - 18), cx + (r - 18), cy + (r - 18), fill=g, outline="")

        for i in range(0, 360, 30):
            self.canvas.create_arc(cx - 120, cy - 120, cx + 120, cy + 120, start=i + self.rotation_angle, extent=5,
                                   outline=c, width=2)

        self.canvas.create_text(cx, cy + 170, text=self.status_text, fill="#ffffff", font=("Consolas", 12, "bold"))

    def animate_hud(self):
        self.rotation_angle = (self.rotation_angle + 1.5) % 360
        if self.status_text == "SPEAKING...":
            self.speaking_pulse += 1
        else:
            self.speaking_pulse = max(0, self.speaking_pulse - 1)
        if self.pulse_growing:
            self.pulse_size += 0.6
            if self.pulse_size > 14: self.pulse_growing = False
        else:
            self.pulse_size -= 0.6
            if self.pulse_size < 0: self.pulse_growing = True
        self.draw_hud()
        self.after(30, self.animate_hud)

    # --- FIXED SPEAK FUNCTION (No more mp3 crashes) ---
    def speak(self, text):
        if not text or len(text.strip()) == 0:
            return  # Don't try to play empty silence

        self.status_text = "SPEAKING..."

        def _speak_thread():
            try:
                voice = VOICES.get(self.current_lang, VOICES["en"])
                communicate = edge_tts.Communicate(text, voice)

                # Use unique filename to avoid "file in use" errors
                filename = f"voice_{int(time.time())}.mp3"
                asyncio.run(communicate.save(filename))

                # Verify file was actually created and has content
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    pygame.mixer.init()
                    pygame.mixer.music.load(filename)
                    pygame.mixer.music.play()

                    while pygame.mixer.music.get_busy():
                        self.speaking_pulse = 6
                        pygame.time.Clock().tick(10)

                    pygame.mixer.quit()
                    # Clean up
                    try:
                        os.remove(filename)
                    except:
                        pass
                else:
                    print("TTS Error: Audio file empty.")

            except Exception as e:
                print(f"Speech Error: {e}")

            self.status_text = "STANDBY"

        threading.Thread(target=_speak_thread, daemon=True).start()

    def listen_loop(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
            self.status_text = "LISTENING..."
            try:
                lang_code = "en-IN"
                if self.current_lang == "hi":
                    lang_code = "hi-IN"
                elif self.current_lang == "gu":
                    lang_code = "gu-IN"

                audio = self.recognizer.listen(source, timeout=20, phrase_time_limit=60)
                command = self.recognizer.recognize_google(audio, language=lang_code).lower()

                if "optimus" in command or self.current_lang != "en":
                    task = command.replace("optimus", "").strip()
                    if task:
                        self.process_command(task)
                    else:
                        self.speak("Yes sir?")
                        self.status_text = "AWAITING..."
                        time.sleep(2.5)
                        self.status_text = "LISTENING..."
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                        audio_2 = self.recognizer.listen(source, timeout=20, phrase_time_limit=60)
                        task_2 = self.recognizer.recognize_google(audio_2, language=lang_code).lower()
                        self.process_command(task_2)
            except:
                pass
            self.is_listening = False
            self.status_text = "STANDBY"

    def process_command(self, command):
        self.status_text = "PROCESSING..."
        response = self.brain.decide_action(command, self.current_lang)
        self.speak(response)

    def toggle_listening_event(self, event=None):
        if not self.is_listening:
            self.is_listening = True
            threading.Thread(target=self.listen_loop, daemon=True).start()


if __name__ == "__main__":
    app = OptimusUI()
    app.mainloop()