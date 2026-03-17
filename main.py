"""
Optimus V3 — Main Orchestrator
Transformers Universe Multi-Agent AI Assistant
"""
import asyncio, threading, time, os, re, json
import tkinter as tk
import customtkinter as ctk
import speech_recognition as sr
import edge_tts, pygame
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from huggingface_hub import InferenceClient

load_dotenv()

from state  import AgentState
from agents import ChatAgent, BrowserAgent, CodeAgent, MemoryAgent, ReminderAgent
from tools.registry import get_tool_specs_text
from ui.hud import (draw_agent, HUD_PALETTES, ASSETS_DIR)

# ── Voice config ──
LANG_CODES = {"en": "en-IN", "hi": "hi-IN", "gu": "gu-IN"}

AGENT_VOICES = {
    "chat":     "en-GB-RyanNeural",    # Optimus   — deep, British
    "browser":  "en-US-AndrewNeural",  # Bumblebee — younger, faster
    "code":     "en-GB-ThomasNeural",  # Wheeljack — technical
    "memory":   "en-GB-RyanNeural",    # Perceptor — same base, slower delivery
    "reminder": "en-US-GuyNeural",     # Ironhide  — gruff
}

LANG_VOICES = {
    "hi": "hi-IN-MadhurNeural",
    "gu": "gu-IN-NiranjanNeural",
}

WAKE_WORDS     = ["optimus", "optimum", "optimas", "hey optimus"]
ACTIVE_TIMEOUT = 45
MODEL          = "Qwen/Qwen2.5-72B-Instruct"


# ================================================================
# SUPERVISOR — LLM routes command to correct agent
# ================================================================
class Supervisor:
    def __init__(self, browser_agent: BrowserAgent):
        self._client        = None
        self._browser_agent = browser_agent

    def _get_client(self):
        if self._client is None:
            self._client = InferenceClient(token=os.getenv("HF_TOKEN"))
        return self._client

    def route(self, state: AgentState) -> AgentState:
        command = state["command"]
        cmd_low = command.lower()

        # ── Hard rules — no LLM needed for these ──
        # Memory recall
        if any(w in cmd_low for w in ["do you remember", "recall", "what did we",
                                       "last time", "previously", "you told me",
                                       "i told you", "we talked"]):
            return {**state, "active_agent": "memory",
                    "tool_name": "memory_recall",
                    "tool_args": {"query": command}}

        # Memory save
        if any(w in cmd_low for w in ["remember this", "save this", "note that",
                                       "don't forget this", "memory stats"]):
            return {**state, "active_agent": "memory",
                    "tool_name": "memory_store",
                    "tool_args": {"text": command, "category": "general"}}

        # Reminder
        if any(w in cmd_low for w in ["remind me", "set a reminder", "reminder",
                                       "alert me", "notify me", "don't let me forget",
                                       "list reminders", "show reminders"]):
            return {**state, "active_agent": "reminder",
                    "tool_name": "set_reminder", "tool_args": {}}

        # ── App opening — always chat agent regardless of browser state ──
        if any(w in cmd_low for w in ["open spotify", "open vs code", "open code",
                                       "open pycharm", "open whatsapp", "open discord",
                                       "open telegram", "open notepad", "open calculator",
                                       "open vlc", "open settings", "open task manager",
                                       "open file explorer", "open explorer",
                                       "open anaconda", "open jupyter", "open terminal",
                                       "open powershell", "open word", "open excel",
                                       "open powerpoint", "open outlook"]):
            return {**state, "active_agent": "chat",
                    "tool_name": "open_app", "tool_args": {"name": cmd_low}}

        # Browser — open or already open
        if self._browser_agent.is_open():
            if not any(w in cmd_low for w in ["open app", "open spotify",
                                               "open vs code", "pause music"]):
                return {**state, "active_agent": "browser",
                        "tool_name": "browser_action", "tool_args": {}}

        if any(w in cmd_low for w in ["open chrome", "open brave", "open edge",
                                       "open firefox", "launch browser",
                                       "close browser", "close chrome", "close brave"]):
            # "open brave/chrome" = launch app via AppOpener, not browser agent
            if any(w in cmd_low for w in ["open chrome", "open brave",
                                           "open edge", "open firefox", "launch browser"]):
                return {**state, "active_agent": "chat",
                        "tool_name": "open_app",
                        "tool_args": {"name": cmd_low.replace("open ", "").replace("launch ", "")}}
            # close browser = browser agent
            return {**state, "active_agent": "browser",
                    "tool_name": "browser_action", "tool_args": {}}

        # Code
        if any(w in cmd_low for w in ["write code", "write a", "create a script",
                                       "build a", "debug", "fix this code",
                                       "explain this code", "run this",
                                       "inject code", "python", "javascript",
                                       "make a function", "create an api"]):
            return {**state, "active_agent": "code",
                    "tool_name": "generate_code", "tool_args": {}}

        # ── LLM supervisor for everything else ──
        prompt = f"""You are the Optimus supervisor. Route this command to the right agent.

Command: "{command}"

Agents:
- chat:    general conversation, web search, open apps, play music, media control
- browser: anything involving a website, browser, search online, watch video
- code:    write code, debug, explain code, run script, programming tasks
- memory:  remember, recall, save, what did we talk about
- reminder: remind me, set alarm, alert at time, note down with time

Respond ONLY with JSON:
{{"agent": "agent_name"}}"""

        try:
            resp = self._get_client().chat_completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30, temperature=0.1
            )
            raw    = resp.choices[0].message.content.strip()
            raw    = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(raw)
            agent  = result.get("agent", "chat")
        except:
            agent  = "chat"

        print(f"[Supervisor] Routed '{command[:40]}...' → {agent}")
        return {**state, "active_agent": agent,
                "tool_name": "", "tool_args": {}}


# ================================================================
# MAIN APP
# ================================================================
ctk.set_appearance_mode("Dark")

class OptimusApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.overrideredirect(True)
        self.geometry("440x640")
        self.config(background="#000001")
        self.attributes("-transparentcolor", "#000001", "-topmost", True)
        self.bind("<Button-1>", self._drag_start)
        self.bind("<B1-Motion>", self._drag_move)

        # ── State ──
        self.status_text   = "STANDBY_EN"
        self.current_lang  = "en"
        self.is_processing = False
        self.stop_speaking = False
        self.pulse         = 0
        self.pulse_dir     = 1
        self._current_agent = "chat"   # tracks which character to show

        # ── Agents ──
        self.memory_agent  = MemoryAgent()
        self.browser_agent = BrowserAgent()
        self.code_agent    = CodeAgent()
        self.reminder_agent = ReminderAgent()
        self.chat_agent    = ChatAgent()
        self.supervisor    = Supervisor(self.browser_agent)

        # Wire reminder speak
        self.reminder_agent.set_speak(self.speak, self)

        # ── Build LangGraph ──
        self._build_graph()

        # ── UI ──
        self.canvas = tk.Canvas(self, width=440, height=520,
                                bg="#000001", highlightthickness=0)
        self.canvas.pack()

        self.status_label = ctk.CTkLabel(
            self, text="STANDBY", font=("Consolas", 11, "bold"),
            text_color="#00eaff", fg_color="transparent"
        )
        self.status_label.pack(pady=2)

        self.lang_frame = ctk.CTkFrame(self, fg_color="#0a0a0a", corner_radius=20)
        self.lang_frame.pack(pady=6)
        self.lang_btns = {}
        for label, code in [("ENG","en"),("HIN","hi"),("GUJ","gu")]:
            btn = ctk.CTkButton(
                self.lang_frame, text=label, width=65,
                font=("Consolas", 11, "bold"),
                command=lambda c=code: self._set_lang(c)
            )
            btn.pack(side="left", padx=6)
            self.lang_btns[code] = btn
        self._update_lang_buttons()

        self._animate()
        threading.Thread(target=self._listen_loop, daemon=True).start()

    # ── Drag ──
    def _drag_start(self, e): self._dx, self._dy = e.x, e.y
    def _drag_move(self, e):
        self.geometry(f"+{self.winfo_x()+e.x-self._dx}+{self.winfo_y()+e.y-self._dy}")

    def _set_lang(self, code):
        self.current_lang = code
        self.status_text  = f"STANDBY_{code.upper()}"
        self._update_lang_buttons()

    def _update_lang_buttons(self):
        colors = {"en": "#00eaff", "hi": "#ff9900", "gu": "#00ff9c"}
        for code, btn in self.lang_btns.items():
            if code == self.current_lang:
                btn.configure(fg_color=colors[code], text_color="#000000")
            else:
                btn.configure(fg_color="#222222", text_color="#888888")

    # ── LangGraph ──
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("supervisor", self.supervisor.route)
        workflow.add_node("chat",       self.chat_agent.run)
        workflow.add_node("browser",    self.browser_agent.run)
        workflow.add_node("code",       self.code_agent.run)
        workflow.add_node("memory",     self.memory_agent.run)
        workflow.add_node("reminder",   self.reminder_agent.run)
        workflow.set_entry_point("supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            lambda s: s["active_agent"],
            {"chat":    "chat",    "browser": "browser",
             "code":    "code",    "memory":  "memory",
             "reminder":"reminder"}
        )
        for node in ["chat", "browser", "code", "memory", "reminder"]:
            workflow.add_edge(node, END)
        self.graph = workflow.compile()

    # ── Animation ──
    def _animate(self):
        self.pulse += self.pulse_dir
        if self.pulse >= 8: self.pulse_dir = -1
        if self.pulse <= 0: self.pulse_dir =  1

        palette   = HUD_PALETTES.get(self.status_text, HUD_PALETTES["STANDBY_EN"])
        pulse_val = self.pulse if self.status_text in (
            "LISTENING", "SPEAKING", "REMEMBERING",
            "BROWSING", "CODING", "REMINDER"
        ) else 0

        self.canvas.delete("all")
        draw_agent(self._current_agent, self.canvas, 220, 260, palette, pulse_val)

        display = self.status_text.replace("STANDBY_", "").replace("_", " ")
        self.status_label.configure(text=display, text_color=palette["primary"])
        self.after(60, self._animate)

    def _switch_character(self, agent: str):
        """Swap to the right Transformer character."""
        if agent != self._current_agent:
            self._current_agent = agent

    # ── Processing ──
    def _process(self, command: str):
        if self.is_processing:
            return
        self.is_processing = True

        # Set status based on likely agent
        cmd = command.lower()
        if any(w in cmd for w in ["do you remember", "recall", "remember this",
                                    "save this", "memory"]):
            self.status_text = "REMEMBERING"
        elif any(w in cmd for w in ["remind me", "reminder", "alert me"]):
            self.status_text = "REMINDER"
        elif any(w in cmd for w in ["open chrome", "open brave", "go to",
                                     "search on", "open browser"]) or self.browser_agent.is_open():
            self.status_text = "BROWSING"
        elif any(w in cmd for w in ["write code", "debug", "script", "function",
                                     "python", "javascript", "build a"]):
            self.status_text = "CODING"
        else:
            self.status_text = "PROCESSING"

        def _run():
            try:
                # Inject memory context if recall keywords present
                mem_ctx = ""
                if any(w in cmd for w in ["do you remember", "recall",
                                           "what did we", "last time"]):
                    mem_ctx = self.memory_agent.recall(command, top_k=3)

                initial_state: AgentState = {
                    "command":        command,
                    "language":       self.current_lang,
                    "active_agent":   "",
                    "response":       "",
                    "tool_name":      "",
                    "tool_args":      {},
                    "memory_context": mem_ctx,
                    "error":          "",
                }
                result = self.graph.invoke(initial_state, {"recursion_limit": 10})
                agent  = result.get("active_agent", "chat")
                reply  = result.get("response", "")

                # Safety — always reset if no reply
                if not reply or not reply.strip():
                    self.is_processing = False
                    self.status_text = f"STANDBY_{self.current_lang.upper()}"
                    return

                # Switch character
                try:
                    self.after(0, lambda a=agent: self._switch_character(a))
                except Exception:
                    pass

                # Speak first — don't wait for memory store
                self.speak(reply, agent=agent)

                # Auto-save to memory in background AFTER speaking starts
                from agents.memory_agent import memory_category
                threading.Thread(
                    target=self.memory_agent.store,
                    args=(command, reply, memory_category(command)),
                    daemon=True
                ).start()

            except Exception as e:
                print(f"[Orchestrator] Error: {e}")
                self.is_processing = False
                self.status_text = f"STANDBY_{self.current_lang.upper()}"

        threading.Thread(target=_run, daemon=True).start()

    # ── TTS ──
    def speak(self, text: str, agent: str = "chat"):
        if not text or not text.strip():
            self.is_processing = False
            self.status_text   = f"STANDBY_{self.current_lang.upper()}"
            return

        self.status_text   = "SPEAKING"
        self.stop_speaking = False

        # Pick voice — non-English overrides agent voice
        if self.current_lang in LANG_VOICES:
            voice = LANG_VOICES[self.current_lang]
        else:
            voice = AGENT_VOICES.get(agent, AGENT_VOICES["chat"])

        def _tts():
            try:
                fname = f"optimus_{int(time.time())}.mp3"
                asyncio.run(edge_tts.Communicate(text, voice).save(fname))
                if os.path.exists(fname) and os.path.getsize(fname) > 0:
                    pygame.mixer.init()
                    pygame.mixer.music.load(fname)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        if self.stop_speaking:
                            pygame.mixer.music.stop()
                            break
                        time.sleep(0.1)
                    pygame.mixer.quit()
                    try: os.remove(fname)
                    except: pass
            except Exception as e:
                print(f"[TTS] Error: {e}")

            # Return to Optimus after speaking
            try:
                self.after(0, lambda: self._switch_character("chat"))
            except Exception:
                pass
            self.is_processing = False
            self.status_text   = f"STANDBY_{self.current_lang.upper()}"

        threading.Thread(target=_tts, daemon=True).start()

    # ── Wake word listener ──
    def _listen_loop(self):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold         = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold          = 0.8

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("[Optimus] Wake word mode — say 'Optimus' to activate.")

            active       = False
            active_until = 0
            greeted      = False

            while True:
                if self.is_processing or self.status_text == "SPEAKING":
                    time.sleep(0.2)
                    continue

                # Expire active window
                if active and time.time() > active_until:
                    active = False
                    self.status_text = f"STANDBY_{self.current_lang.upper()}"
                    print("[Optimus] Active window expired.")
                    self.speak("Going standby, sir.")

                try:
                    plimit = 8 if active else 4
                    tout   = None if not active else ACTIVE_TIMEOUT
                    audio  = recognizer.listen(source, timeout=tout,
                                               phrase_time_limit=plimit)

                    # Transcribe
                    best_text, best_lang = None, self.current_lang
                    for lang in ([self.current_lang] +
                                 [l for l in ["en","hi","gu"] if l != self.current_lang]):
                        try:
                            text = recognizer.recognize_google(
                                audio, language=LANG_CODES[lang])
                            if text:
                                best_text, best_lang = text.lower(), lang
                                break
                        except: continue

                    if not best_text:
                        continue

                    is_wake = any(w in best_text for w in WAKE_WORDS)

                    if not active:
                        if is_wake:
                            active       = True
                            active_until = time.time() + ACTIVE_TIMEOUT
                            self.status_text = "LISTENING"
                            print("[Optimus] Activated.")
                            if not greeted:
                                greeted = True
                                self.speak("Optimus online. How can I help you, sir?")
                            else:
                                self.speak("Yes sir?")
                        continue

                    # Active window
                    active_until = time.time() + ACTIVE_TIMEOUT

                    if is_wake and len(best_text.split()) <= 3:
                        self.speak("I'm listening.")
                        continue

                    if best_lang != self.current_lang:
                        self.current_lang = best_lang
                        try:
                            self.after(0, self._update_lang_buttons)
                        except Exception:
                            pass

                    print(f"[Heard] ({best_lang}) {best_text}")
                    self._process(best_text)

                except sr.WaitTimeoutError:
                    if active:
                        active = False
                        self.status_text = f"STANDBY_{self.current_lang.upper()}"
                        self.speak("Going standby.")
                except Exception as e:
                    print(f"[Listen] Error: {e}")
                    time.sleep(0.5)


if __name__ == "__main__":
    pygame.mixer.pre_init(44100, -16, 2, 512)
    app = OptimusApp()
    app.mainloop()