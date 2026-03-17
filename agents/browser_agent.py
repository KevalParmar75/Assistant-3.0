"""
Browser Agent — Bumblebee
Attaches to your existing Chrome/Brave via remote debugging.
Opens a new tab — never touches your profile lock.
"""
import re, json, threading, subprocess, time, os
from state import AgentState

CHARACTER = "bumblebee"
VOICE     = "en-US-AndrewNeural"
MODEL     = "Qwen/Qwen2.5-72B-Instruct"
DEBUG_PORT = 9222


class BrowserAgent:
    def __init__(self):
        self._playwright = None
        self._browser    = None
        self._page       = None
        self._lock       = threading.Lock()
        self._client     = None
        print("[Bumblebee] Browser agent ready.")

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=os.getenv("HF_TOKEN"))
        return self._client

    def is_open(self) -> bool:
        try:
            return (self._page is not None and
                    not self._page.is_closed())
        except:
            return False

    def _launch_browser_with_debug(self):
        """Launch Chrome/Brave with remote debugging port if not already open."""
        username = os.environ.get("USERNAME", "User")
        brave_exe   = rf"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
        chrome_exe  = rf"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
        chrome_exe2 = rf"C:\Program Files\Google\Chrome\Application\chrome.exe"

        if os.path.exists(brave_exe):   exe = brave_exe
        elif os.path.exists(chrome_exe): exe = chrome_exe
        elif os.path.exists(chrome_exe2): exe = chrome_exe2
        else:
            print("[Bumblebee] No browser found!")
            return False

        # Launch with remote debugging — uses YOUR existing profile, no lock issues
        subprocess.Popen([
            exe,
            f"--remote-debugging-port={DEBUG_PORT}",
            "--no-first-run",
            "--no-default-browser-check",
        ])
        time.sleep(2)  # wait for browser to start
        return True

    def _ensure_browser(self):
        from playwright.sync_api import sync_playwright
        if self._playwright is None:
            self._playwright = sync_playwright().start()

        if self._browser is None:
            # Try connecting to existing browser first
            try:
                self._browser = self._playwright.chromium.connect_over_cdp(
                    f"http://localhost:{DEBUG_PORT}"
                )
                print("[Bumblebee] Connected to existing browser.")
            except Exception:
                # Browser not running with debug port — launch it
                print("[Bumblebee] Launching browser with debug port...")
                if not self._launch_browser_with_debug():
                    return
                try:
                    self._browser = self._playwright.chromium.connect_over_cdp(
                        f"http://localhost:{DEBUG_PORT}"
                    )
                    print("[Bumblebee] Connected.")
                except Exception as e:
                    print(f"[Bumblebee] Could not connect: {e}")
                    return

        # Open a new tab
        try:
            ctx = self._browser.contexts[0] if self._browser.contexts else self._browser.new_context()
            self._page = ctx.new_page()
            print("[Bumblebee] New tab opened.")
        except Exception as e:
            print(f"[Bumblebee] Tab error: {e}")
            self._browser = None
            self._page    = None

    def close(self):
        try:
            if self._page and not self._page.is_closed():
                self._page.close()
        except: pass
        finally:
            self._page = None

    def get_page_context(self) -> str:
        if not self.is_open():
            return "No browser tab open."
        try:
            url   = self._page.url
            title = self._page.title()
            text  = self._page.inner_text("body")[:1500].replace("\n", " ").strip()
            return f"URL: {url}\nTitle: {title}\nContent: {text}"
        except:
            return f"URL: {self._page.url if self._page else 'unknown'}"

    def execute_plan(self, command: str) -> str:
        with self._lock:
            try:
                self._ensure_browser()
            except Exception as e:
                print(f"[Bumblebee] Browser setup failed: {e}")
                return "Couldn't open the browser, sir."

            if not self._page:
                return "Browser tab couldn't be opened."

            page_ctx = self.get_page_context()
            client   = self._get_client()

            plan_prompt = f"""You are Bumblebee, a browser automation agent using Playwright.

Current browser state:
{page_ctx}

User command: "{command}"

IMPORTANT: The user may speak in Hindi or Gujarati. Always translate their intent to English for URLs and search queries.

Produce a JSON array of steps. Each step:
  {{"action": "...", "value": "...", "description": "..."}}

Actions:
- navigate: go to URL directly
- click_nth: click the Nth result/video (value = number)
- click: click by visible text
- type: type text (value = "css_selector|||text")
- press: keyboard key
- scroll: scroll pixels
- wait: wait ms
- done: finished (description = what to say to user)

RULES:
- YouTube search → navigate to https://www.youtube.com/results?search_query=query+here
- Google search  → navigate to https://www.google.com/search?q=query+here
- URL encode spaces as +
- Prefer direct URL navigation over typing
- Always end with "done"

Respond ONLY with valid JSON array."""

            try:
                resp = self._get_client().chat_completion(
                    model=MODEL,
                    messages=[{"role": "user", "content": plan_prompt}],
                    max_tokens=600, temperature=0.2
                )
                raw  = resp.choices[0].message.content.strip()
                raw  = re.sub(r"```json|```", "", raw).strip()
                plan = json.loads(raw)
                print(f"[Bumblebee] Plan: {json.dumps(plan, indent=2)}")
            except Exception as e:
                print(f"[Bumblebee] Plan failed: {e}")
                return "Couldn't figure out the steps for that."

            last_desc = "Done."
            for step in plan:
                action = step.get("action", "")
                value  = str(step.get("value", ""))
                desc   = step.get("description", "")
                print(f"[Bumblebee] {action} → {value}")
                try:
                    if action == "navigate":
                        self._page.goto(value, wait_until="domcontentloaded", timeout=20000)
                        self._page.wait_for_timeout(2000)

                    elif action == "click":
                        try:
                            self._page.get_by_text(value, exact=False).first.click(timeout=5000)
                        except:
                            self._page.click(value, timeout=5000)
                        self._page.wait_for_timeout(1000)

                    elif action == "click_nth":
                        n = int(value) - 1
                        clicked = False
                        for sel in ["ytd-video-renderer #video-title",
                                    "ytd-video-renderer a#video-title",
                                    "a#video-title-link",
                                    "ytd-rich-item-renderer #video-title"]:
                            items = self._page.query_selector_all(sel)
                            if items and n < len(items):
                                items[n].scroll_into_view_if_needed()
                                items[n].click()
                                clicked = True
                                break
                        if not clicked:
                            links = [l for l in self._page.query_selector_all("a[href]")
                                     if l.is_visible()]
                            if n < len(links):
                                links[n].click()
                        self._page.wait_for_timeout(1500)

                    elif action == "type":
                        if "|||" in value:
                            sel, text = value.split("|||", 1)
                            self._page.fill(sel.strip(), text.strip())
                        else:
                            self._page.keyboard.type(value)
                        self._page.wait_for_timeout(400)

                    elif action == "press":
                        self._page.keyboard.press(value)
                        self._page.wait_for_timeout(1200)

                    elif action == "scroll":
                        px = int(value) if value.lstrip("-").isdigit() else 500
                        self._page.mouse.wheel(0, px)
                        self._page.wait_for_timeout(500)

                    elif action == "wait":
                        ms = int(value) if value.isdigit() else 1500
                        self._page.wait_for_timeout(ms)

                    elif action == "done":
                        last_desc = desc or "Done."
                        break

                except Exception as e:
                    print(f"[Bumblebee] Step failed ({action}={value}): {e}")
                    continue

            return last_desc

    def run(self, state: AgentState) -> AgentState:
        command = state["command"]
        if any(w in command.lower() for w in
               ["close browser", "close tab", "close chrome", "close brave"]):
            self.close()
            return {**state, "response": "Tab closed.", "active_agent": "browser"}
        response = self.execute_plan(command)
        return {**state, "response": response, "active_agent": "browser"}


CHARACTER = "bumblebee"
VOICE     = "en-US-AndrewNeural"   # younger, faster
MODEL     = "Qwen/Qwen2.5-72B-Instruct"


class BrowserAgent:
    def __init__(self):
        self._playwright = None
        self._browser    = None
        self._page       = None
        self._lock       = threading.Lock()
        self._client     = None
        print("[Bumblebee] Browser agent ready.")

    def _get_client(self):
        if self._client is None:
            import os
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=os.getenv("HF_TOKEN"))
        return self._client

    def is_open(self) -> bool:
        try:
            return (self._browser is not None and
                    self._page is not None and
                    not self._page.is_closed())
        except:
            return False

    def _ensure_browser(self):
        from playwright.sync_api import sync_playwright
        import os, shutil, tempfile
        if self._playwright is None:
            self._playwright = sync_playwright().start()
        if self._browser is None or not self._browser.is_connected():
            username = os.environ.get("USERNAME", "User")

            brave_exe   = rf"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
            chrome_exe  = rf"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
            chrome_exe2 = rf"C:\Program Files\Google\Chrome\Application\chrome.exe"

            brave_profile  = rf"C:\Users\{username}\AppData\Local\BraveSoftware\Brave-Browser\User Data"
            chrome_profile = rf"C:\Users\{username}\AppData\Local\Google\Chrome\User Data"

            # Pick executable
            if os.path.exists(brave_exe):
                exe     = brave_exe
                profile = brave_profile
            elif os.path.exists(chrome_exe):
                exe     = chrome_exe
                profile = chrome_profile
            elif os.path.exists(chrome_exe2):
                exe     = chrome_exe2
                profile = chrome_profile
            else:
                exe = profile = None

            if exe and profile and os.path.exists(profile):
                # Copy just the Default profile to a temp dir — avoids lock conflict
                tmp_dir     = os.path.join(tempfile.gettempdir(), "optimus_browser_profile")
                tmp_default = os.path.join(tmp_dir, "Default")
                src_default = os.path.join(profile, "Default")

                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir, exist_ok=True)
                    if os.path.exists(src_default):
                        print("[Bumblebee] Copying browser profile (first run)...")
                        shutil.copytree(src_default, tmp_default,
                                        ignore=shutil.ignore_patterns(
                                            "*.log","*Lock*","lockfile",
                                            "Cache*","Code Cache","GPUCache"
                                        ))
                        print("[Bumblebee] Profile copied.")

                try:
                    self._browser = self._playwright.chromium.launch_persistent_context(
                        user_data_dir=tmp_dir,
                        executable_path=exe,
                        headless=False,
                        args=["--start-maximized", "--no-first-run",
                              "--disable-sync", "--no-default-browser-check"],
                        no_viewport=True,
                    )
                    self._page = self._browser.pages[0] if self._browser.pages else self._browser.new_page()
                    print(f"[Bumblebee] Browser launched with profile.")
                    return
                except Exception as e:
                    print(f"[Bumblebee] Profile launch failed: {e} — trying fresh browser.")

            # Fallback — fresh browser no profile
            print("[Bumblebee] Launching fresh browser (no profile).")
            browser = self._playwright.chromium.launch(
                headless=False,
                channel="chrome" if (os.path.exists(chrome_exe) or os.path.exists(chrome_exe2)) else "chromium",
                args=["--start-maximized"]
            )
            self._browser = browser
            ctx = browser.new_context(no_viewport=True)
            self._page = ctx.new_page()

        if self._page is None or self._page.is_closed():
            try:
                self._page = self._browser.new_page()
            except:
                self._page = self._browser.pages[0] if self._browser.pages else None

    def close(self):
        try:
            if self._page:        self._page.close()
            if self._browser:     self._browser.close()
            if self._playwright:  self._playwright.stop()
        except: pass
        finally:
            self._playwright = None
            self._browser    = None
            self._page       = None

    def get_page_context(self) -> str:
        if not self.is_open():
            return "No browser open."
        try:
            url   = self._page.url
            title = self._page.title()
            text  = self._page.inner_text("body")[:1500].replace("\n", " ").strip()
            return f"URL: {url}\nTitle: {title}\nContent: {text}"
        except:
            return f"URL: {self._page.url if self._page else 'unknown'}"

    def execute_plan(self, command: str) -> str:
        with self._lock:
            try:
                self._ensure_browser()
            except Exception as e:
                print(f"[Bumblebee] Browser launch failed: {e}")
                return "Couldn't launch the browser, sir."

            page_ctx = self.get_page_context()
            client   = self._get_client()

            plan_prompt = f"""You are Bumblebee, a browser automation agent using Playwright.

Current browser state:
{page_ctx}

User command: "{command}"

IMPORTANT: The user may speak in Hindi or Gujarati. Always translate their intent to English for URLs and search queries.

Produce a JSON array of steps. Each step:
  {{"action": "...", "value": "...", "description": "..."}}

Actions available:
- navigate: go to a URL directly
- click_nth: click the Nth result/video (value = number as string)
- click: click element by visible text
- type: type text (value = "css_selector|||text_to_type")
- press: keyboard key (Enter, Escape, ArrowDown etc.)
- scroll: scroll pixels (value = "500" or "-500")
- wait: wait milliseconds
- done: finished (description = confirmation message to speak)

RULES:
- YouTube search → navigate to https://www.youtube.com/results?search_query=query+here
- Google search  → navigate to https://www.google.com/search?q=query+here
- Always URL-encode spaces as + in search queries
- Prefer direct URL navigation over typing in boxes
- Always end with "done" action

Respond ONLY with valid JSON array. No markdown, no explanation."""

            try:
                resp = client.chat_completion(
                    model=MODEL,
                    messages=[{"role": "user", "content": plan_prompt}],
                    max_tokens=600, temperature=0.2
                )
                raw  = resp.choices[0].message.content.strip()
                raw  = re.sub(r"```json|```", "", raw).strip()
                plan = json.loads(raw)
                print(f"[Bumblebee] Plan: {json.dumps(plan, indent=2)}")
            except Exception as e:
                print(f"[Bumblebee] Plan failed: {e}")
                return "Couldn't figure out the steps for that one."

            last_desc = "Done."
            for step in plan:
                action = step.get("action", "")
                value  = str(step.get("value", ""))
                desc   = step.get("description", "")
                print(f"[Bumblebee] {action} → {value}")
                try:
                    if action == "navigate":
                        self._page.goto(value, wait_until="domcontentloaded", timeout=20000)
                        self._page.wait_for_timeout(2000)

                    elif action == "click":
                        try:
                            self._page.get_by_text(value, exact=False).first.click(timeout=5000)
                        except:
                            self._page.click(value, timeout=5000)
                        self._page.wait_for_timeout(1000)

                    elif action == "click_nth":
                        n       = int(value) - 1
                        clicked = False
                        for sel in ["ytd-video-renderer #video-title",
                                    "ytd-video-renderer a#video-title",
                                    "a#video-title-link",
                                    "ytd-rich-item-renderer #video-title"]:
                            items = self._page.query_selector_all(sel)
                            if items and n < len(items):
                                items[n].scroll_into_view_if_needed()
                                items[n].click()
                                clicked = True
                                break
                        if not clicked:
                            links = [l for l in self._page.query_selector_all("a[href]")
                                     if l.is_visible()]
                            if n < len(links):
                                links[n].click()
                        self._page.wait_for_timeout(1500)

                    elif action == "type":
                        if "|||" in value:
                            sel, text = value.split("|||", 1)
                            self._page.fill(sel.strip(), text.strip())
                        else:
                            self._page.keyboard.type(value)
                        self._page.wait_for_timeout(400)

                    elif action == "press":
                        self._page.keyboard.press(value)
                        self._page.wait_for_timeout(1200)

                    elif action == "scroll":
                        px = int(value) if value.lstrip("-").isdigit() else 500
                        self._page.mouse.wheel(0, px)
                        self._page.wait_for_timeout(500)

                    elif action == "wait":
                        ms = int(value) if value.isdigit() else 1500
                        self._page.wait_for_timeout(ms)

                    elif action == "done":
                        last_desc = desc or "Done."
                        break

                except Exception as e:
                    print(f"[Bumblebee] Step failed ({action}={value}): {e}")
                    continue

            return last_desc

    # ── LangGraph node ──
    def run(self, state: AgentState) -> AgentState:
        command = state["command"]
        if any(w in command.lower() for w in
               ["close browser", "close chrome", "close brave", "close edge"]):
            self.close()
            return {**state, "response": "Browser closed.", "active_agent": "browser"}
        response = self.execute_plan(command)
        return {**state, "response": response, "active_agent": "browser"}