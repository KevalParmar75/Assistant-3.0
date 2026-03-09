import asyncio
import threading
import tkinter as tk
import customtkinter as ctk
import speech_recognition as sr
import pywhatkit
import edge_tts
import pygame
import os
import time
import json
import datetime
import webbrowser
import pyautogui
from typing import TypedDict
from langgraph.graph import StateGraph, END
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS
from AppOpener import open as open_app
from dotenv import load_dotenv

# ── Memory imports ──
import chromadb
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage

load_dotenv()

# ================= CONFIGURATION =================
HF_TOKEN    = os.getenv("HF_TOKEN")
BRAIN_MODEL = "moonshotai/Kimi-K2-Thinking"

# ── Memory paths — E:\optimus\memory\ ──
MEMORY_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")
CHROMA_DIR     = os.path.join(MEMORY_DIR, "chromadb")
LLAMAINDEX_DIR = os.path.join(MEMORY_DIR, "llamaindex")
RAW_LOG        = os.path.join(MEMORY_DIR, "conversation_log.jsonl")

os.makedirs(CHROMA_DIR,     exist_ok=True)
os.makedirs(LLAMAINDEX_DIR, exist_ok=True)

VOICES = {
    "en": "en-GB-RyanNeural",
    "hi": "hi-IN-MadhurNeural",
    "gu": "gu-IN-NiranjanNeural"
}

LANG_CODES = {
    "en": "en-IN",
    "hi": "hi-IN",
    "gu": "gu-IN"
}

HUD_PALETTES = {
    "STANDBY_EN":  {"primary": "#00eaff", "secondary": "#0077aa", "dark": "#003344", "visor": "#00eaff", "glow": "#004466"},
    "STANDBY_HI":  {"primary": "#ff9900", "secondary": "#cc6600", "dark": "#442200", "visor": "#ff9900", "glow": "#663300"},
    "STANDBY_GU":  {"primary": "#00ff9c", "secondary": "#00aa66", "dark": "#003322", "visor": "#00ff9c", "glow": "#004433"},
    "LISTENING":   {"primary": "#ff2d2d", "secondary": "#aa0000", "dark": "#330000", "visor": "#ff2d2d", "glow": "#550000"},
    "PROCESSING":  {"primary": "#ffd500", "secondary": "#aa8800", "dark": "#443300", "visor": "#ffd500", "glow": "#665500"},
    "SPEAKING":    {"primary": "#00ff9c", "secondary": "#00aa66", "dark": "#003322", "visor": "#00ff9c", "glow": "#004433"},
    "REMEMBERING": {"primary": "#cc44ff", "secondary": "#880099", "dark": "#220033", "visor": "#cc44ff", "glow": "#440066"},
}

# ── Pixel art — extracted from reference image ──
_HEAD_GRID = [
    "1111111111111111111111111111111111111111", # 0
    "1111111111111111111111111111111111111111", # 1
    "1115311111111111111111111111111111131111", # 2
    "1115318111111111111111111111111118131111", # 3
    "1115318111111111154331111111111118131111", # 4
    "1115318111111111111221111111111118131111", # 5
    "1115318111111111153112211111111118131111", # 6
    "1135318111111111135333211111111118131111", # 7
    "1125328111111115551111312111111118131111", # 8
    "1125325111115555188888832222111118131111", # 9
    "1125325115555555581111832222221118131111", # 10
    "1125325115555544581111811222222118231111", # 11
    "1155115155544112111111181112222218111811", # 12
    "1154118154312111111111182111122218111811", # 13
    "1153111541555441811111181222221121111811", # 14
    "1152155354333335811111181222222211111811", # 15
    "1152533433333331811111118222222221111811", # 16
    "1151433433333318111111118122222221111811", # 17
    "1151433433333311111111118122222221111811", # 18
    "1151433433333318111111118122222221111811", # 19
    "1141433433333338818888111222222221111111", # 20
    "1151431333333331811111188222222221111811", # 21
    "1151431333333333811111111222222221111811", # 22
    "1151431333333333181881882222222221111811", # 23
    "1131431333333333381881112222222221111111", # 24
    "1581331333333333388881822222222221111811", # 25
    "5581411333333332281881812222222221121111", # 26
    "4381181883333333218888112222232181811111", # 27
    "3381188858111113388888822111118888811111", # 28
    "3381188811221118181881818111121118811811", # 29
    "3381188881611166181181811611161118811811", # 30  EYES
    "3381188181812111818888111111111118811811", # 31
    "3381158881881111188888111111111118811111", # 32
    "3318181118881888888888188881188111818111", # 33
    "3381858818888888811888188888888818888811", # 34
    "2388888817877778718888188888818818111811", # 35
    "2288888818777777771881881888881818188821", # 36
    "1111188818777777877888888888881818811111", # 37
    "1111188817777778577888888888888888811111", # 38
    "1111188815777777777118888888888888811111", # 39
    "1111111118777777777888888888888811111111", # 40
    "1111118111777777777888888888881818111111", # 41
    "1111118118187777777888888888118818111111", # 42
    "1111118118888775777888888888888118111111", # 43
    "1111118118818775577888888881181818111111", # 44
    "1111118118851588777888881881888118111111", # 45
    "1111111111818888887888888881181811111111", # 46
    "1111188818881188877888818811188188811111", # 47
    "1111111111111111111111111111111111111111", # 48
]

_COLOR_MAP = {
    "1": None,        # transparent
    "2": "#081428",   # very dark navy
    "3": "#102a6e",   # dark blue
    "4": "#1a4898",   # mid blue
    "5": "#2d6fc0",   # lighter blue
    "6": "#00ccff",   # cyan eyes (dynamic)
    "7": "#c8d8e8",   # silver faceplate
    "8": "#606878",   # grey panels
}

def draw_optimus_head(canvas, cx, cy, palette, pulse=0):
    PS   = 10
    COLS = len(_HEAD_GRID[0])
    ROWS = len(_HEAD_GRID)
    ox   = cx - (COLS * PS) // 2
    oy   = cy - (ROWS * PS) // 2
    visor_color = palette["visor"]
    glow_color  = palette["glow"]
    for row_idx, row in enumerate(_HEAD_GRID):
        for col_idx, cell in enumerate(row):
            if cell == "1":
                continue
            x0 = ox + col_idx * PS
            y0 = oy + row_idx * PS
            x1 = x0 + PS
            y1 = y0 + PS
            if cell == "6":
                if pulse > 0:
                    canvas.create_rectangle(x0-2, y0-2, x1+2, y1+2,
                                            fill=glow_color, outline="")
                fill = visor_color
            else:
                fill = _COLOR_MAP.get(cell)
                if fill is None:
                    continue
            canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="")


# =================================================================
# DUAL MEMORY SYSTEM
# ChromaDB   → general conversations, preferences, contacts
# LlamaIndex → code sessions, posts, structured content
# =================================================================
class OptimusMemory:
    def __init__(self):
        self._embedder      = None
        self._chroma_client = None
        self._chroma_col    = None
        self._llama_index   = None
        print("[Memory] Ready — lazy loading on first use.")

    @property
    def embedder(self):
        if self._embedder is None:
            print("[Memory] Loading sentence-transformers model...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("[Memory] Embedding model loaded.")
        return self._embedder

    @property
    def chroma(self):
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
            self._chroma_col    = self._chroma_client.get_or_create_collection(
                name="optimus_memory",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[Memory] ChromaDB ready — {self._chroma_col.count()} entries.")
        return self._chroma_col

    @property
    def llama(self):
        if self._llama_index is None:
            try:
                storage = StorageContext.from_defaults(persist_dir=LLAMAINDEX_DIR)
                self._llama_index = load_index_from_storage(storage)
                print("[Memory] LlamaIndex loaded from disk.")
            except Exception:
                self._llama_index = VectorStoreIndex([])
                print("[Memory] LlamaIndex — fresh index created.")
        return self._llama_index

    # ── Store exchange — runs in background thread ──
    def store(self, user_text: str, bot_text: str, category: str = "general"):
        timestamp = datetime.datetime.now().isoformat()
        entry = {"timestamp": timestamp, "user": user_text, "bot": bot_text, "category": category}
        try:
            with open(RAW_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Memory] Log error: {e}")
        threading.Thread(target=self._embed_and_store,
                         args=(user_text, bot_text, category, timestamp),
                         daemon=True).start()

    def _embed_and_store(self, user_text, bot_text, category, timestamp):
        chunk = f"User: {user_text}\nOptimus: {bot_text}"
        try:
            if category in ("code", "post"):
                doc = Document(text=chunk, metadata={"timestamp": timestamp, "category": category})
                self.llama.insert(doc)
                self.llama.storage_context.persist(persist_dir=LLAMAINDEX_DIR)
            else:
                emb    = self.embedder.encode(chunk).tolist()
                doc_id = f"mem_{timestamp.replace(':', '-').replace('.', '-')}"
                self.chroma.add(ids=[doc_id], embeddings=[emb],
                                documents=[chunk],
                                metadatas=[{"timestamp": timestamp, "category": category}])
        except Exception as e:
            print(f"[Memory] Embed/store error: {e}")

    # ── Recall — triggered only when user says remember/recall ──
    def recall(self, query: str, category: str = "general", top_k: int = 4) -> str:
        try:
            if category in ("code", "post"):
                nodes = self.llama.as_retriever(similarity_top_k=top_k).retrieve(query)
                return "\n---\n".join([n.get_content() for n in nodes]) if nodes else ""
            else:
                if self.chroma.count() == 0:
                    return ""
                emb     = self.embedder.encode(query).tolist()
                results = self.chroma.query(
                    query_embeddings=[emb],
                    n_results=min(top_k, self.chroma.count())
                )
                docs = results.get("documents", [[]])[0]
                return "\n---\n".join(docs) if docs else ""
        except Exception as e:
            print(f"[Memory] Recall error: {e}")
            return ""

    def stats(self) -> str:
        try:
            c = self.chroma.count()
            l = len(self.llama.docstore.docs)
            return f"General: {c} entries | Code/Post: {l} entries"
        except:
            return "Stats unavailable."


# ── Shared global memory instance ──
MEMORY = OptimusMemory()


# ── Category detection ──
def _memory_category(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["code", "script", "function", "class", "debug", "error",
                             "fix", "flask", "django", "python", "javascript", "html",
                             "css", "api", "program", "build app", "write app"]):
        return "code"
    if any(w in t for w in ["post", "linkedin", "twitter", "tweet", "caption",
                             "blog", "article", "announcement", "content", "write about"]):
        return "post"
    return "general"


# ── Recall intent detection ──
def _wants_recall(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ["remember", "recall", "do you remember", "what did we",
                                 "last time", "previously", "earlier we", "before you",
                                 "we talked", "you told me", "i told you", "history"])


# ================= LANGGRAPH STATE =================
class AgentState(TypedDict):
    current_command: str
    next_step:       str
    response_text:   str
    language:        str
    memory_context:  str


# ================= AGENT =================
class OptimusAgent:
    def __init__(self):
        self.client = InferenceClient(token=HF_TOKEN)

    def _ask(self, messages: list, max_tokens: int = 256) -> str:
        try:
            response = self.client.chat_completion(
                model=BRAIN_MODEL, messages=messages,
                max_tokens=max_tokens, temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[HF Error] {e}")
            return "I ran into an issue, sir."

    def router(self, state: AgentState) -> AgentState:
        cmd = state["current_command"].lower()
        # Memory triggers — check first
        if _wants_recall(cmd):
            return {**state, "next_step": "recall"}
        if any(w in cmd for w in ["remember this", "save this", "remember that",
                                   "save that", "don't forget", "note that",
                                   "memory stats", "how many memories"]):
            return {**state, "next_step": "memory_save"}
        if any(w in cmd for w in ["search", "who is", "what is", "tell me about", "look up"]):
            return {**state, "next_step": "search"}
        if any(w in cmd for w in ["skip", "next song", "pause", "play music", "stop music"]):
            return {**state, "next_step": "media"}
        if "whatsapp" in cmd:
            return {**state, "next_step": "whatsapp"}
        if any(w in cmd for w in ["open", "launch", "start"]):
            return {**state, "next_step": "open_app"}
        if any(w in cmd for w in ["play", "song", "music", "youtube"]):
            return {**state, "next_step": "play"}
        return {**state, "next_step": "chat"}

    def chat_node(self, state: AgentState) -> AgentState:
        lang_map = {"en": "English", "hi": "Hindi", "gu": "Gujarati"}
        lang     = lang_map.get(state["language"], "English")
        cmd      = state["current_command"]
        system   = (f"You are Optimus, a sharp and witty AI assistant. "
                    f"Reply only in {lang}. Keep it short and punchy — 1-3 sentences max.")
        mem = state.get("memory_context", "")
        if mem:
            system += f"\n\n[Relevant past context]:\n{mem}"
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": cmd}]
        response = self._ask(messages)
        MEMORY.store(cmd, response, _memory_category(cmd))
        return {**state, "response_text": response}

    def recall_node(self, state: AgentState) -> AgentState:
        cmd      = state["current_command"]
        lang_map = {"en": "English", "hi": "Hindi", "gu": "Gujarati"}
        lang     = lang_map.get(state["language"], "English")
        category = _memory_category(cmd)
        context  = MEMORY.recall(cmd, category=category, top_k=4)
        if not context:
            return {**state, "response_text": "I don't have any memory of that yet, sir."}
        messages = [
            {"role": "system", "content": (
                f"You are Optimus. The user wants you to recall something. "
                f"Reply only in {lang}. Be concise and natural — like you genuinely remember."
            )},
            {"role": "user", "content": (
                f"User asked: '{cmd}'\n\nRelevant past conversations:\n{context}\n\n"
                f"Summarize what's relevant in 2-3 sentences."
            )}
        ]
        return {**state, "response_text": self._ask(messages, max_tokens=300)}

    def memory_save_node(self, state: AgentState) -> AgentState:
        cmd = state["current_command"].lower()
        # memory stats query
        if "stats" in cmd or "how many" in cmd:
            return {**state, "response_text": f"Memory status — {MEMORY.stats()}"}
        # Save the last exchange (or the command itself as a note)
        ui = getattr(self, "_ui_ref", None)
        last_user = getattr(ui, "_last_user", "") if ui else ""
        last_bot  = getattr(ui, "_last_bot",  "") if ui else ""
        if last_user and last_bot:
            category = _memory_category(last_user)
            MEMORY.store(last_user, last_bot, category)
            return {**state, "response_text": f"Got it — I've saved that to {'code/post memory' if category in ('code','post') else 'general memory'}."}
        # Fallback — save the command itself as a note
        MEMORY.store(cmd, "[User note]", "general")
        return {**state, "response_text": "Noted and saved to memory, sir."}

    def tool_node(self, state: AgentState) -> AgentState:
        action = state["next_step"]
        cmd    = state["current_command"]

        if action == "search":
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(cmd, max_results=2))
                raw = " ".join([r['body'] for r in results]) if results else "No results."
            except Exception as e:
                raw = f"Search failed: {e}"
            messages = [
                {"role": "system", "content": "You are Optimus. Summarize briefly in 1-2 sentences. No bullet points."},
                {"role": "user",   "content": f"User asked: '{cmd}'. Result: {raw}"}
            ]
            response = self._ask(messages)
            MEMORY.store(cmd, response, "general")          # ✅ auto-save
            return {**state, "response_text": response}

        elif action == "media":
            if   "skip" in cmd or "next" in cmd:  pyautogui.press("nexttrack")
            elif "pause" in cmd or "stop" in cmd: pyautogui.press("playpause")
            else:                                  pyautogui.press("playpause")
            MEMORY.store(cmd, "Media control executed.", "general")   # ✅ auto-save
            return {**state, "response_text": "Done."}

        elif action == "open_app":
            messages = [
                {"role": "system", "content": "Extract only the app name. Reply with just the name."},
                {"role": "user",   "content": cmd}
            ]
            app_name = self._ask(messages, max_tokens=20).lower().strip()
            try:
                if "youtube" in app_name: webbrowser.open("https://www.youtube.com")
                else: open_app(app_name, match_closest=True, output=False)
                response = f"Opening {app_name}."
            except:
                response = f"Couldn't open {app_name}."
            MEMORY.store(cmd, response, "general")          # ✅ auto-save
            return {**state, "response_text": response}

        elif action == "play":
            messages = [
                {"role": "system", "content": "Extract only the song/video name for YouTube search. Just the query."},
                {"role": "user",   "content": cmd}
            ]
            query = self._ask(messages, max_tokens=30).strip()
            try:
                pywhatkit.playonyt(query)
                response = f"Playing {query}."
            except:
                response = "Couldn't play that."
            MEMORY.store(cmd, response, "general")          # ✅ auto-save
            return {**state, "response_text": response}

        elif action == "whatsapp":
            return {**state, "response_text": "WhatsApp integration coming soon, sir."}

        return {**state, "response_text": "Done."}


# ================= LANGUAGE DETECTOR =================
def detect_language(text: str) -> str:
    gujarati = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')
    hindi    = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    if gujarati > 0: return "gu"
    if hindi    > 0: return "hi"
    return "en"


# ================= UI =================
ctk.set_appearance_mode("Dark")

class OptimusV2(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.overrideredirect(True)
        self.geometry("440x620")
        self.config(background='#000001')
        self.attributes('-transparentcolor', '#000001', '-topmost', True)
        self.bind("<Button-1>", self.start_move)
        self.bind("<B1-Motion>", self.do_move)

        self.status_text   = "STANDBY_EN"
        self.current_lang  = "en"
        self.is_processing = False
        self.pulse         = 0
        self.pulse_dir     = 1
        self.stop_speaking = False
        self._last_user    = ""
        self._last_bot     = ""

        self.setup_graph()

        self.canvas = tk.Canvas(self, width=440, height=510,
                                bg='#000001', highlightthickness=0)
        self.canvas.pack()

        self.status_label = ctk.CTkLabel(self, text="STANDBY",
                                          font=("Consolas", 11, "bold"),
                                          text_color="#00eaff", fg_color="transparent")
        self.status_label.pack(pady=2)

        self.lang_frame = ctk.CTkFrame(self, fg_color="#0a0a0a", corner_radius=20)
        self.lang_frame.pack(pady=8)
        self.lang_btns = {}
        for label, code in [("ENG", "en"), ("HIN", "hi"), ("GUJ", "gu")]:
            btn = ctk.CTkButton(self.lang_frame, text=label, width=65,
                                font=("Consolas", 11, "bold"),
                                command=lambda c=code: self.set_lang(c))
            btn.pack(side="left", padx=6)
            self.lang_btns[code] = btn
        self.update_lang_buttons()

        self.animate_hud()
        threading.Thread(target=self.always_on_listen, daemon=True).start()

    def start_move(self, event): self.x, self.y = event.x, event.y
    def do_move(self, event):
        self.geometry(f"+{self.winfo_x()+event.x-self.x}+{self.winfo_y()+event.y-self.y}")

    def set_lang(self, code):
        self.current_lang = code
        self.status_text  = f"STANDBY_{code.upper()}"
        self.update_lang_buttons()

    def update_lang_buttons(self):
        colors = {"en": "#00eaff", "hi": "#ff9900", "gu": "#00ff9c"}
        for code, btn in self.lang_btns.items():
            if code == self.current_lang:
                btn.configure(fg_color=colors[code], text_color="#000000")
            else:
                btn.configure(fg_color="#222222", text_color="#888888")

    def setup_graph(self):
        agent    = OptimusAgent()
        agent._ui_ref = self          # give agent a ref so memory_save can grab last exchange
        workflow = StateGraph(AgentState)
        workflow.add_node("router",      agent.router)
        workflow.add_node("chat",        agent.chat_node)
        workflow.add_node("tools",       agent.tool_node)
        workflow.add_node("recall",      agent.recall_node)
        workflow.add_node("memory_save", agent.memory_save_node)
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            lambda state: state["next_step"],
            {"chat": "chat", "recall": "recall", "memory_save": "memory_save",
             "search": "tools", "media": "tools",
             "open_app": "tools", "play": "tools", "whatsapp": "tools"}
        )
        workflow.add_edge("chat",        END)
        workflow.add_edge("recall",      END)
        workflow.add_edge("memory_save", END)
        workflow.add_edge("tools",       END)
        self.app_graph = workflow.compile()

    def animate_hud(self):
        self.pulse += self.pulse_dir
        if self.pulse >= 8:  self.pulse_dir = -1
        if self.pulse <= 0:  self.pulse_dir =  1

        palette = HUD_PALETTES.get(self.status_text, HUD_PALETTES["STANDBY_EN"])
        self.canvas.delete("all")
        pulse_val = self.pulse if self.status_text in ("LISTENING", "SPEAKING", "REMEMBERING") else 0
        draw_optimus_head(self.canvas, cx=220, cy=255, palette=palette, pulse=pulse_val)

        display = self.status_text.replace("STANDBY_", "").replace("_", " ")
        self.status_label.configure(text=display, text_color=palette["primary"])
        self.after(60, self.animate_hud)

    def always_on_listen(self):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold         = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold          = 0.8

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("[Optimus] Always-on listening started.")

            while True:
                if self.is_processing or self.status_text == "SPEAKING":
                    time.sleep(0.2)
                    continue
                try:
                    self.status_text = f"STANDBY_{self.current_lang.upper()}"
                    audio = recognizer.listen(source, timeout=None, phrase_time_limit=12)

                    best_text, best_lang = None, self.current_lang
                    lang_order = [self.current_lang] + [l for l in ["en","hi","gu"] if l != self.current_lang]

                    for lang in lang_order:
                        try:
                            text = recognizer.recognize_google(audio, language=LANG_CODES[lang])
                            if text:
                                detected = detect_language(text)
                                if detected == lang or (detected == "en" and lang == "en"):
                                    best_text, best_lang = text.lower(), lang
                                    break
                                elif best_text is None:
                                    best_text, best_lang = text.lower(), lang
                        except:
                            continue

                    if best_text:
                        if best_lang != self.current_lang:
                            print(f"[Auto-lang] Switched to {best_lang}")
                            self.current_lang = best_lang
                            self.after(0, self.update_lang_buttons)
                        print(f"[Heard] ({best_lang}) {best_text}")
                        self.process_command(best_text)

                except sr.WaitTimeoutError:
                    pass
                except Exception as e:
                    print(f"[Listen Error] {e}")
                    time.sleep(0.5)

    def process_command(self, command):
        if self.is_processing:
            return
        self.is_processing = True
        self.status_text = "REMEMBERING" if (_wants_recall(command) or
                           any(w in command.lower() for w in ["remember this","save this","remember that","save that","don't forget","note that"])) else "PROCESSING"

        def _run():
            try:
                res = self.app_graph.invoke(
                    {"current_command": command, "next_step": "",
                     "response_text": "", "language": self.current_lang,
                     "memory_context": ""},
                    {"recursion_limit": 10}
                )
                reply = res["response_text"]
                # Track last exchange so memory_save can access it
                self._last_user = command
                self._last_bot  = reply
                self.speak(reply)
            except Exception as e:
                print(f"[Graph Error] {e}")
                self.is_processing = False
                self.status_text = f"STANDBY_{self.current_lang.upper()}"

        threading.Thread(target=_run, daemon=True).start()

    def speak(self, text):
        if not text or not text.strip():
            self.is_processing = False
            self.status_text = f"STANDBY_{self.current_lang.upper()}"
            return
        self.status_text   = "SPEAKING"
        self.stop_speaking = False

        def _thread():
            try:
                fname = f"optimus_{int(time.time())}.mp3"
                asyncio.run(edge_tts.Communicate(text, VOICES[self.current_lang]).save(fname))
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
                print(f"[TTS Error] {e}")
            self.is_processing = False
            self.status_text = f"STANDBY_{self.current_lang.upper()}"

        threading.Thread(target=_thread, daemon=True).start()


if __name__ == "__main__":
    app = OptimusV2()
    app.mainloop()