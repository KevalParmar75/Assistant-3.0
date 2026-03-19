"""
Memory Agent — Perceptor
ChromaDB (general) + LlamaIndex (code/posts)
No LLM needed — pure vector search.
"""
import os, json, datetime, threading
from state import AgentState

# ── Paths ──
MEMORY_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory")
CHROMA_DIR     = os.path.join(MEMORY_DIR, "chromadb")
LLAMAINDEX_DIR = os.path.join(MEMORY_DIR, "llamaindex")
RAW_LOG        = os.path.join(MEMORY_DIR, "conversation_log.jsonl")
os.makedirs(CHROMA_DIR,     exist_ok=True)
os.makedirs(LLAMAINDEX_DIR, exist_ok=True)

CHARACTER = "perceptor"
VOICE     = "en-GB-RyanNeural"   # slower, thoughtful tone


class MemoryAgent:
    def __init__(self):
        self._embedder      = None
        self._chroma_client = None
        self._chroma_col    = None
        self._llama_index   = None
        self._embedder_ready = False
        self._embed_lock     = threading.Lock()
        self._wire_tools()
        print("[Perceptor] Memory agent ready.")
        # Pre-warm embedder in background so first command never blocks
        threading.Thread(target=self._prewarm, daemon=True).start()

    def _prewarm(self):
        try:
            from sentence_transformers import SentenceTransformer
            with self._embed_lock:
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                self._embedder_ready = True
            _ = self.chroma
            print("[Perceptor] Embedder and ChromaDB ready.")
        except Exception as e:
            print(f"[Perceptor] Pre-warm error: {e}")

    # ── Lazy loaders ──
    @property
    def embedder(self):
        with self._embed_lock:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                self._embedder_ready = True
            return self._embedder

    @property
    def chroma(self):
        if self._chroma_client is None:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
            self._chroma_col    = self._chroma_client.get_or_create_collection(
                name="optimus_memory", metadata={"hnsw:space": "cosine"}
            )
        return self._chroma_col

    @property
    def llama(self):
        if self._llama_index is None:
            from llama_index.core import (VectorStoreIndex, StorageContext,
                                          load_index_from_storage)
            try:
                storage = StorageContext.from_defaults(persist_dir=LLAMAINDEX_DIR)
                self._llama_index = load_index_from_storage(storage)
            except Exception:
                from llama_index.core import VectorStoreIndex
                self._llama_index = VectorStoreIndex([])
        return self._llama_index

    # ── Wire real implementations into tools registry ──
    def _wire_tools(self):
        from tools.registry import REGISTRY
        REGISTRY["memory_store"]  = self.store_tool
        REGISTRY["memory_recall"] = self.recall_tool

    # ── Store (auto-save every exchange) ──
    def store(self, user_text: str, bot_text: str, category: str = "general"):
        timestamp = datetime.datetime.now().isoformat()
        entry = {"timestamp": timestamp, "user": user_text,
                 "bot": bot_text, "category": category}
        try:
            with open(RAW_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Perceptor] Log error: {e}")
        threading.Thread(target=self._embed,
                         args=(user_text, bot_text, category, timestamp),
                         daemon=True).start()

    def _embed(self, user_text, bot_text, category, timestamp):
        # Skip embedding silently if model not ready yet — raw log already saved
        if not self._embedder_ready:
            return
        from llama_index.core import Document
        chunk = f"User: {user_text}\nOptimus: {bot_text}"
        try:
            if category in ("code", "post"):
                doc = Document(text=chunk,
                               metadata={"timestamp": timestamp, "category": category})
                self.llama.insert(doc)
                self.llama.storage_context.persist(persist_dir=LLAMAINDEX_DIR)
            else:
                emb    = self.embedder.encode(chunk).tolist()
                doc_id = f"mem_{timestamp.replace(':', '-').replace('.', '-')}"
                self.chroma.add(ids=[doc_id], embeddings=[emb],
                                documents=[chunk],
                                metadatas=[{"timestamp": timestamp, "category": category}])
        except Exception as e:
            print(f"[Perceptor] Embed error: {e}")

    # ── Recall ──
    def recall(self, query: str, category: str = "general", top_k: int = 4) -> str:
        if not self._embedder_ready:
            return ""
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
            print(f"[Perceptor] Recall error: {e}")
            return ""

    def stats(self) -> str:
        try:
            c = self.chroma.count()
            l = len(self.llama.docstore.docs)
            return f"General: {c} entries | Code/Post: {l} entries"
        except:
            return "Stats unavailable."

    # ── Tool wrappers ──
    def store_tool(self, text: str, category: str = "general") -> str:
        self.store(text, "[manual save]", category)
        return f"Saved to {category} memory."

    def recall_tool(self, query: str) -> str:
        result = self.recall(query)
        return result if result else "Nothing found in memory."

    # ── LangGraph node ──
    def run(self, state: AgentState) -> AgentState:
        from tools.registry import call_tool
        import os, datetime

        tool_name = state.get("tool_name", "memory_recall")
        command   = state.get("command", "")

        # ── Notes ──
        if tool_name == "note":
            notes_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memory", "notes.txt"
            )
            cmd = command.lower()
            # Read
            if any(w in cmd for w in ["read", "show", "what are", "list"]):
                if os.path.exists(notes_file):
                    with open(notes_file, "r", encoding="utf-8") as f:
                        notes = f.read().strip()
                    result = f"Your notes: {notes}" if notes else "No notes saved yet."
                else:
                    result = "No notes saved yet."
            else:
                # Save
                note_text = command
                for phrase in ["take a note", "note down", "write this down",
                               "add a note", "save a note", "jot down", "note that"]:
                    note_text = note_text.lower().replace(phrase, "").strip()
                if note_text:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    with open(notes_file, "a", encoding="utf-8") as f:
                        f.write(f"[{ts}] {note_text}\n")
                    self.store(command, f"Note saved: {note_text}", "general")
                    result = f"Got it. Note saved."
                else:
                    result = "What would you like me to note down?"
            return {**state, "response": result, "active_agent": "memory"}

        tool_args = state.get("tool_args", {"query": command})
        result    = call_tool(tool_name, tool_args)
        return {**state, "response": result, "active_agent": "memory"}


# ── Category helper (used by other agents) ──
def memory_category(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["code", "script", "function", "class", "debug",
                              "fix", "flask", "django", "python", "javascript",
                              "html", "css", "api", "program"]):
        return "code"
    if any(w in t for w in ["post", "linkedin", "twitter", "tweet", "caption",
                              "blog", "article", "announcement", "content"]):
        return "post"
    return "general"