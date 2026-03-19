"""
Vision Agent — Optimus's Eyes
Uses Qwen2-VL to see the screen and interact with it.
Screenshots are in-memory only — never written to disk.
"""
import os, base64, io, time, json, re
import pyautogui
from state import AgentState

CHARACTER = "optimus"
MODEL     = "Qwen/Qwen2.5-VL-7B-Instruct"

pyautogui.FAILSAFE = False


class VisionAgent:
    def __init__(self):
        self._client      = None
        self._screen_w    = None
        self._screen_h    = None
        self._get_screen_size()
        print("[Vision] Vision agent ready.")

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=os.getenv("HF_TOKEN"))
        return self._client

    def _get_screen_size(self):
        try:
            self._screen_w, self._screen_h = pyautogui.size()
            print(f"[Vision] Screen: {self._screen_w}x{self._screen_h}")
        except:
            self._screen_w, self._screen_h = 1920, 1080

    def capture(self) -> str:
        """Take screenshot → resize to 1280x720 → return as base64 PNG string."""
        import mss
        from PIL import Image
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[0]  # full screen
                raw     = sct.grab(monitor)
                img     = Image.frombytes("RGB", raw.size, raw.rgb)

            # Resize to 1280x720 — saves API tokens, VL still reads fine
            img = img.resize((1280, 720), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return b64
        except Exception as e:
            print(f"[Vision] Capture failed: {e}")
            return ""

    def _scale_coords(self, x: int, y: int) -> tuple[int, int]:
        """Scale VL coords (1280x720) back to actual screen resolution."""
        sx = self._screen_w / 1280
        sy = self._screen_h / 720
        return int(x * sx), int(y * sy)

    def ask_vl(self, b64_image: str, prompt: str) -> str:
        """Send screenshot + prompt to Qwen VL, return raw response."""
        try:
            resp = self._get_client().chat_completion(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }],
                max_tokens=300,
                temperature=0.1
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Vision] VL API failed: {e}")
            return ""

    def describe(self) -> str:
        """Describe what's currently on screen."""
        b64 = self.capture()
        if not b64:
            return "Couldn't capture the screen."
        prompt = """Describe what you see on this screen in 2-3 sentences.
Be specific — mention the app, website, content visible, and any important UI elements.
Keep it concise and natural, like you're telling someone what's on their screen."""
        result = self.ask_vl(b64, prompt)
        return result if result else "I can see your screen but couldn't describe it."

    def find_and_act(self, command: str) -> str:
        """
        Screenshot → VL finds element → pyautogui acts on it.
        Returns spoken confirmation.
        """
        b64 = self.capture()
        if not b64:
            return "Couldn't capture the screen, sir."

        prompt = f"""You are controlling a computer. Look at this screenshot carefully.

User wants to: "{command}"

Find the exact element to interact with and determine the action.

Respond ONLY with valid JSON (no markdown):
{{
  "action": "click" | "double_click" | "right_click" | "type" | "scroll_up" | "scroll_down" | "key" | "describe",
  "x": <x coordinate 0-1280, required for click actions>,
  "y": <y coordinate 0-720, required for click actions>,
  "value": "<text to type OR key name like 'enter','escape','space' OR scroll amount>",
  "description": "<what you found and what you will do, spoken naturally>"
}}

Rules:
- For click: provide exact x,y of the CENTER of the element
- For scroll: no x,y needed, value = number of scroll clicks (e.g. "3")
- For key press: no x,y needed, value = key name
- For type: provide x,y of input field to click first, value = text to type
- If you cannot find the element, set action to "describe" and explain what you see
- Be precise with coordinates — click the center of buttons/links"""

        raw = self.ask_vl(b64, prompt)
        if not raw:
            return "I looked at the screen but couldn't figure out what to do."

        try:
            raw_clean = re.sub(r"```json|```", "", raw).strip()
            result    = json.loads(raw_clean)
        except:
            print(f"[Vision] VL response not JSON: {raw}")
            # VL returned text description — just speak it
            return raw

        action = result.get("action", "")
        x      = result.get("x", 0)
        y      = result.get("y", 0)
        value  = str(result.get("value", ""))
        desc   = result.get("description", "Done.")

        print(f"[Vision] Action: {action} at ({x},{y}) value={value}")
        print(f"[Vision] VL says: {desc}")

        try:
            if action == "click":
                rx, ry = self._scale_coords(x, y)
                pyautogui.click(rx, ry)
                time.sleep(0.5)

            elif action == "double_click":
                rx, ry = self._scale_coords(x, y)
                pyautogui.doubleClick(rx, ry)
                time.sleep(0.5)

            elif action == "right_click":
                rx, ry = self._scale_coords(x, y)
                pyautogui.rightClick(rx, ry)
                time.sleep(0.5)

            elif action == "type":
                # Click the field first, then type
                if x and y:
                    rx, ry = self._scale_coords(x, y)
                    pyautogui.click(rx, ry)
                    time.sleep(0.3)
                pyautogui.typewrite(value, interval=0.05)

            elif action == "scroll_down":
                clicks = int(value) if value.isdigit() else 3
                pyautogui.scroll(-clicks)

            elif action == "scroll_up":
                clicks = int(value) if value.isdigit() else 3
                pyautogui.scroll(clicks)

            elif action == "key":
                pyautogui.press(value)
                time.sleep(0.3)

            elif action == "describe":
                return desc

        except Exception as e:
            print(f"[Vision] Action failed: {e}")
            return f"I found it but couldn't interact with it: {e}"

        return desc

    def run(self, state: AgentState) -> AgentState:
        command = state["command"].lower()

        # Pure description — no action needed
        if any(w in command for w in ["what's on screen", "what is on screen",
                                       "what do you see", "describe screen",
                                       "what's open", "what's on my screen"]):
            response = self.describe()
        else:
            # Vision-guided action
            response = self.find_and_act(state["command"])

        return {**state, "response": response, "active_agent": "vision"}