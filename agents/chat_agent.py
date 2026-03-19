"""
Chat Agent — Optimus Prime
General conversation, web search, app control, media.
The default agent — handles everything not claimed by specialists.
"""
import os
from state import AgentState
from tools.registry import call_tool, get_tool_specs_text

CHARACTER = "optimus"   # pixel art — not a GIF
VOICE     = "en-GB-RyanNeural"
MODEL     = "Qwen/Qwen2.5-72B-Instruct"

# Tools this agent is allowed to use
ALLOWED_TOOLS = ["web_search", "open_app", "open_url",
                 "media_control", "play_youtube",
                 "memory_store", "memory_recall"]


class ChatAgent:
    def __init__(self):
        self._client = None
        print("[Optimus] Chat agent online.")

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=os.getenv("HF_TOKEN"))
        return self._client

    def _ask(self, messages: list, max_tokens: int = 256) -> str:
        try:
            resp = self._get_client().chat_completion(
                model=MODEL, messages=messages,
                max_tokens=max_tokens, temperature=0.7
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Optimus] LLM error: {e}")
            return "I ran into an issue, sir."

    def decide_tool(self, command: str) -> dict | None:
        """Ask LLM if a tool is needed and which one."""
        allowed_specs = "\n".join([
            line for line in get_tool_specs_text().split("\n")
            if any(t in line for t in ALLOWED_TOOLS)
        ])
        prompt = f"""You are Optimus. A user said: "{command}"

Available tools:
{allowed_specs}

If a tool is needed, respond with JSON:
{{"tool": "tool_name", "args": {{...}}}}

If no tool needed (just conversation), respond with:
{{"tool": null, "args": {{}}}}

Respond ONLY with JSON. No explanation."""
        try:
            raw = self._ask([{"role": "user", "content": prompt}], max_tokens=100)
            import re, json
            raw = re.sub(r"```json|```", "", raw).strip()
            return json.loads(raw)
        except:
            return {"tool": None, "args": {}}

    # ── LangGraph node ──
    def run(self, state: AgentState) -> AgentState:
        # Direct response — no LLM needed (e.g. time/date)
        if state.get("tool_name") == "direct":
            return {**state,
                    "response": state.get("tool_args", {}).get("response", ""),
                    "active_agent": "chat"}

        command  = state["command"]
        language = state.get("language", "en")
        mem_ctx  = state.get("memory_context", "")
        lang_map = {"en": "English", "hi": "Hindi", "gu": "Gujarati"}
        lang     = lang_map.get(language, "English")

        # Check if a tool is needed
        tool_decision = self.decide_tool(command)
        tool_name     = tool_decision.get("tool")
        tool_args     = tool_decision.get("args", {})

        if tool_name and tool_name in ALLOWED_TOOLS:
            tool_result = call_tool(tool_name, tool_args)
            # If tool returned raw data (e.g. web search), summarize it
            if tool_name == "web_search":
                system = (f"You are Optimus. Summarize this search result in 1-2 sentences in {lang}. "
                          f"Be direct, no bullet points.")
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": f"Query: {command}\nResult: {tool_result}"}
                ]
                response = self._ask(messages)
            else:
                response = tool_result
        else:
            # Pure conversation
            system = (f"You are Optimus Prime — sharp, witty, direct. "
                      f"Reply only in {lang}. Keep it 1-3 sentences max.")
            if mem_ctx:
                system += f"\n\n[Past context]:\n{mem_ctx}"
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": command}
            ]
            response = self._ask(messages)

        return {**state, "response": response,
                "active_agent": "chat",
                "tool_name": tool_name or "",
                "tool_args": tool_args}