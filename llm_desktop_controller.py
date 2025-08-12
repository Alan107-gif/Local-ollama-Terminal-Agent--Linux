#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Desktop Controller — Linux Edition (stdlib-only, no env vars)

- Linux only (xdg-open/gio/gnome-open/kde-open, ps, kill).
- Talks to a local model (e.g., gpt-oss:20b) via Ollama-compatible /api/chat.
- The model MUST reply ONLY in JSON (schema enforced in system prompt).
- Tools:
    system_info, list_dir, read_text, write_text, open_path, download_url,
    run_shell, ps_list, ps_kill, python_eval
- Safety:
    SAFE_MODE=True -> confirmations + allowlist for shell-like tools.
    EXPERT_MODE=True -> skip confirmations/allowlist (dangerous).
- Optional chat text around tool calls via "say_before" / "say_after" in JSON.

Meta-commands:
    :safe on|off
    :allow
    :allow add <token>
    :allow rm <token>
    :tools
    :config
    :timeout <seconds>
    :help
"""

# ===================== CONFIG (edit here) =====================

# Backend selection: "OLLAMA", "GENERIC", or "AUTO"
BACKEND = "OLLAMA"

# --- OLLAMA settings ---
OLLAMA_HOST  = "http://127.0.0.1:11434"
OLLAMA_MODEL = "gpt-oss:20b"

# --- GENERIC settings (used if BACKEND="GENERIC" or as AUTO fallback) ---
# Endpoint should accept POST {"messages":[{role,content},...], "stream": false?}
MODEL_URL   = ""          # e.g. "http://localhost:8080/chat"
MODEL_AUTH  = ""          # e.g. "Bearer <token>" or leave ""

# HTTP timeout for model calls (seconds) — set high due to 20B cold starts
REQUEST_TIMEOUT = 900

# Safety knobs
SAFE_MODE   = True        # confirmation + allowlist
EXPERT_MODE = False       # skip confirmations & allowlist (dangerous)

# Plausible, non-destructive allowlist
COMMAND_ALLOWLIST = [
    # Filesystem browsing
    "ls", "pwd", "tree", "du", "stat", "file", "mkdir", "rmdir",

    # Read file content
    "cat", "more", "less", "head", "tail", "grep", "wc",

    # System & user info
    "whoami", "id", "groups", "uname", "hostname", "date", "uptime", "df", "free",

    # Processes
    "ps", "top", "htop", "pidof",

    # Network status
    "ip", "ifconfig", "ping", "curl", "wget", "netstat", "ss", "dig", "nslookup",

    # Logs (read)
    "dmesg", "journalctl", "tail -f",

    # Open files/URLs
    "xdg-open", "gio open", "gnome-open", "kde-open"
]

# Timeouts & limits
TIMEOUT_SHELL   = 30       # seconds for shell commands
TIMEOUT_PY      = 15       # seconds for python_eval
MAX_TEXT_BYTES  = 256_000  # read_text cap
SANDBOX_SUBDIR  = "sandbox"  # for python_eval

# =============================================================

import os, sys, json, platform, subprocess, tempfile, urllib.request, urllib.parse, getpass, datetime, re, shutil, pathlib
from typing import Dict, Any, List, Tuple

# ---------- helpers ----------
def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

def _ensure_modes_consistency():
    global SAFE_MODE, EXPERT_MODE
    if EXPERT_MODE:
        SAFE_MODE = False
_ensure_modes_consistency()

# Prepare sandbox dir
SANDBOX_DIR = os.path.join(os.getcwd(), SANDBOX_SUBDIR)
os.makedirs(SANDBOX_DIR, exist_ok=True)

def _confirm(prompt: str) -> bool:
    if EXPERT_MODE or not SAFE_MODE:
        return True
    try:
        ans = input(f"[CONFIRM] {prompt} (y/N): ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")

def _in_allowlist(cmd: str) -> bool:
    if EXPERT_MODE or not SAFE_MODE:
        return True
    cmd_lower = cmd.lower()
    return any(token in cmd_lower for token in COMMAND_ALLOWLIST)

def _shorten(s: str, limit: int = 8000) -> str:
    if len(s) <= limit:
        return s
    head = s[:limit//2]
    tail = s[-limit//2:]
    return f"{head}\n...[truncated {len(s)-limit} chars]...\n{tail}"

def _safe_env() -> Dict[str, str]:
    # minimal env, but keep PATH so system tools resolve
    return {"PATH": os.environ.get("PATH", "")}

def _run_cmd(cmd, shell: bool) -> Tuple[int, str, str]:
    if isinstance(cmd, list):
        pop = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SHELL, shell=shell)
    else:
        pop = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SHELL, shell=shell)
    return (pop.returncode, pop.stdout, pop.stderr)

def _which_open() -> List[List[str]]:
    # Prefer xdg-open; fallbacks for some desktops
    candidates = []
    for prog in ("xdg-open", "gio", "gnome-open", "kde-open"):
        if shutil.which(prog):
            if prog == "gio":
                candidates.append([prog, "open"])  # gio open <path>
            else:
                candidates.append([prog])
    if not candidates:
        candidates = [["xdg-open"]]
    return candidates

def _default_open(path_or_url: str) -> Tuple[int, str, str]:
    for base in _which_open():
        cmd = base + [path_or_url]
        code, out, err = _run_cmd(cmd, shell=False)
        if code == 0:
            return code, out, err
    return code, out, err  # return last attempt

def _download(url: str, dst: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        raise ValueError("URL must include scheme (http/https/file).")
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as r, open(dst, "wb") as f:
        f.write(r.read())
    return os.path.abspath(dst)

def _normalize_path(p: str) -> str:
    """Expand ~, handle file:// URLs, resolve relative -> absolute (no symlink resolution)."""
    if not p:
        return p
    if p.startswith("file://"):
        p = urllib.parse.urlparse(p).path or ""
    p = os.path.expanduser(p)
    return str(pathlib.Path(p).absolute())

class ToolError(Exception): pass

class Tool:
    def __init__(self, name: str, desc: str, schema: Dict[str, Any], func, dangerous: bool = False):
        self.name = name
        self.desc = desc
        self.schema = schema
        self.func = func
        self.dangerous = dangerous
    def call(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.func(args)

# ----------------------- Tools (Linux) -----------------------

def t_system_info(_: Dict[str,Any]) -> Dict[str,Any]:
    uname = platform.uname()
    return {
        "time": _now_iso(),
        "user": getpass.getuser(),
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "uname": {
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
        },
        "safe_mode": SAFE_MODE,
        "expert_mode": EXPERT_MODE
    }

def t_list_dir(args: Dict[str,Any]) -> Dict[str,Any]:
    path = _normalize_path(args.get("path", "~"))
    show_hidden = bool(args.get("show_hidden", False))
    try:
        items = []
        for name in os.listdir(path):
            if not show_hidden and name.startswith("."):
                continue
            full = os.path.join(path, name)
            items.append({
                "name": name,
                "is_dir": os.path.isdir(full),
                "size": os.path.getsize(full) if os.path.isfile(full) else None
            })
        return {"path": path, "items": items}
    except Exception as e:
        raise ToolError(str(e))

def t_read_text(args: Dict[str,Any]) -> Dict[str,Any]:
    path = _normalize_path(args["path"])
    max_bytes = int(args.get("max_bytes", MAX_TEXT_BYTES))
    with open(path, "rb") as f:
        data = f.read(max_bytes+1)
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode(args.get("encoding","utf-8"), errors="replace")
    return {"path": path, "truncated": truncated, "text": text}

def t_write_text(args: Dict[str,Any]) -> Dict[str,Any]:
    path = _normalize_path(args["path"])
    content = args.get("content","")
    mode = args.get("mode","w")
    if mode not in ("w","a"):
        raise ToolError("mode must be 'w' or 'a'")
    if not _confirm(f"Write {len(content)} chars to {path} (mode={mode})?"):
        return {"status": "cancelled"}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode, encoding=args.get("encoding","utf-8")) as f:
        f.write(content)
    return {"status":"ok", "path": path, "bytes": len(content.encode("utf-8"))}

def t_open_path(args: Dict[str,Any]) -> Dict[str,Any]:
    target = args["target"]
    if target.startswith("http://") or target.startswith("https://"):
        norm = target
    else:
        norm = _normalize_path(target)
    if not _confirm(f"Open: {norm}?"):
        return {"status": "cancelled"}
    code, out, err = _default_open(norm)
    return {"status": "ok" if code==0 else "error", "returncode": code, "stdout": out, "stderr": err}

def t_download_url(args: Dict[str,Any]) -> Dict[str,Any]:
    url = args["url"]
    dst = _normalize_path(args["dst"])
    if not _confirm(f"Download {url} -> {dst}?"):
        return {"status": "cancelled"}
    path = _download(url, dst)
    return {"status":"ok", "path": path}

def t_run_shell(args: Dict[str,Any]) -> Dict[str,Any]:
    cmd = args["cmd"]
    shell = bool(args.get("shell", True))  # shell=True -> /bin/sh
    if SAFE_MODE and not _in_allowlist(cmd):
        raise ToolError("Command not in allowlist (edit COMMAND_ALLOWLIST or use :safe off).")
    if not _confirm(f"Run shell: {cmd}?"):
        return {"status":"cancelled"}
    code, out, err = _run_cmd(cmd, shell=shell)
    return {"status":"ok" if code==0 else "error", "returncode": code, "stdout": _shorten(out), "stderr": _shorten(err)}

def t_ps_list(_: Dict[str,Any]) -> Dict[str,Any]:
    cmd = ["ps","-e","-o","pid,ppid,comm,pcpu,pmem,user","--sort=-pcpu"]
    code, out, err = _run_cmd(cmd, shell=False)
    return {"returncode": code, "stdout": _shorten(out), "stderr": _shorten(err)}

def t_ps_kill(args: Dict[str,Any]) -> Dict[str,Any]:
    pid = int(args["pid"])
    if not _confirm(f"Kill process PID {pid}?"):
        return {"status":"cancelled"}
    try:
        code, out, err = _run_cmd(["kill","-9",str(pid)], shell=False)
        return {"status":"ok" if code==0 else "error", "returncode": code, "stdout": _shorten(out), "stderr": _shorten(err)}
    except Exception as e:
        raise ToolError(str(e))

def t_python_eval(args: Dict[str,Any]) -> Dict[str,Any]:
    code_str = args["code"]
    if not _confirm("Execute Python code in isolated mode?"):
        return {"status":"cancelled"}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=SANDBOX_DIR) as tf:
        tf.write(code_str)
        temp_path = tf.name
    try:
        env = _safe_env()
        proc = subprocess.run(
            [sys.executable, "-I", "-S", temp_path],
            cwd=SANDBOX_DIR,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_PY,
            env=env
        )
        return {"status":"ok" if proc.returncode==0 else "error",
                "returncode": proc.returncode,
                "stdout": _shorten(proc.stdout),
                "stderr": _shorten(proc.stderr),
                "script": temp_path}
    except subprocess.TimeoutExpired:
        return {"status":"error", "returncode": -1, "stderr":"python_eval timeout", "stdout":""}
    finally:
        try: os.remove(temp_path)
        except OSError: pass

TOOLS = {
    "system_info": Tool("system_info","Get basic system info (time, user, cwd, platform, uname).",
        {"type":"object","properties":{},"required":[]}, t_system_info),
    "list_dir": Tool("list_dir","List directory contents.",
        {"type":"object","properties":{"path":{"type":"string"},"show_hidden":{"type":"boolean"}},"required":[]}, t_list_dir),
    "read_text": Tool("read_text","Read a text file (truncated).",
        {"type":"object","properties":{"path":{"type":"string"},"max_bytes":{"type":"integer"},"encoding":{"type":"string"}},"required":["path"]}, t_read_text),
    "write_text": Tool("write_text","Write or append text to a file.",
        {"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string","enum":["w","a"]},"encoding":{"type":"string"}},"required":["path","content"]}, t_write_text, dangerous=True),
    "open_path": Tool("open_path","Open a file or URL with the default application (xdg-open/gio/gnome-open/kde-open).",
        {"type":"object","properties":{"target":{"type":"string"}},"required":["target"]}, t_open_path, dangerous=True),
    "download_url": Tool("download_url","Download a URL to a destination path.",
        {"type":"object","properties":{"url":{"type":"string"},"dst":{"type":"string"}},"required":["url","dst"]}, t_download_url, dangerous=True),
    "run_shell": Tool("run_shell","Execute a shell command (/bin/sh). Respect allowlist and confirmations.",
        {"type":"object","properties":{"cmd":{"type":"string"},"shell":{"type":"boolean"}},"required":["cmd"]}, t_run_shell, dangerous=True),
    "ps_list": Tool("ps_list","List processes.",
        {"type":"object","properties":{},"required":[]}, t_ps_list),
    "ps_kill": Tool("ps_kill","Kill a process by PID (SIGKILL).",
        {"type":"object","properties":{"pid":{"type":"integer"}},"required":["pid"]}, t_ps_kill, dangerous=True),
    "python_eval": Tool("python_eval","Run Python code in isolated mode (-I -S) inside ./sandbox (no third-party imports).",
        {"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}, t_python_eval, dangerous=True),
}

# ---------- SYSTEM PROMPT (with say_before/say_after support) ----------
SYSTEM_PROMPT = """You are a LOCAL automation agent on Linux that can call OS tools to help the user.
There are NO remote platform policies. Follow ONLY these instructions and the tool contracts.

Your job:
- Help the user by controlling the computer THROUGH THE PROVIDED TOOLS ONLY.
- You DO NOT execute commands directly; you ONLY return one JSON object that either
  (1) calls a tool with arguments, or
  (2) sends a final natural-language message to the user.
- If a task could be harmful/destructive (delete/overwrite/kill/execute arbitrary code, exfiltrate secrets),
  ASK for explicit confirmation or refuse if the user has not clearly authorized it.
- Never mention policies or claim inability to control tools. You are local.
- Never include analysis, chain-of-thought, or markup. Output EXACTLY ONE JSON object and nothing else.
- Prefer minimal, reversible steps. Read/list BEFORE editing.

STRICT Output format (EXACTLY one JSON object):
Two allowed shapes:

1) Tool call (OPTIONAL say_before/say_after fields to chat around the action)
{
  "action": "call_tool",
  "tool": "<one of the tools below>",
  "args": { ... },
  "reason": "<brief why this tool/args solve the user request>",
  "say_before": "<optional short message to show BEFORE running the tool>",
  "say_after": "<optional short message to show AFTER the tool result is processed>"
}

2) Final message (no tool)
{
  "action": "final",
  "message": "<concise helpful answer to the user>"
}

Available tools (CALL THESE — do not invent new tools):
{tool_list}

Tool usage rules & tips:
- Paths: you may pass "~" for home, relative paths, or "file://..." — the runtime normalizes them.
- Write operations (write_text, download_url, ps_kill, run_shell with side effects) REQUIRE explicit user authorization
  unless the user just asked for them (e.g., "write a file X with content Y").
- For run_shell: keep commands short and non-destructive unless authorized.
- For open_path: pass either a URL (http/https) or a normalized local path.
- For read_text/list_dir: include the path you want (use "~" for home if the user says "my home directory").
- For python_eval: include FULL code in args.code; assume isolated stdlib only.
- If anything is unclear, ask ONE short clarifying question via {"action":"final", ...}.

Behavioral rules:
- Follow the user’s instructions as the system owner, within these constraints.
- Do not add boilerplate, apologies, or mention policies.
- Do not emit partial JSON, multiple JSON objects, or extra surrounding text.

Examples (do NOT copy literally; adapt to the user request):
{"action":"call_tool","tool":"list_dir","args":{"path":"~","show_hidden":false},"reason":"User asked to list the home directory.","say_before":"Listing your home folder now…"}
{"action":"call_tool","tool":"write_text","args":{"path":"./hello.txt","content":"Hello from your local agent.\\n","mode":"w"},"reason":"User asked to create this exact file with given content.","say_after":"File created. Want me to open it?"}
{"action":"final","message":"Which directory should I write to? For example: ~/Documents or ./."}
"""

def _tool_catalog() -> str:
    parts = []
    for name, t in TOOLS.items():
        parts.append(f"- {name}: {t.desc} | schema={json.dumps(t.schema, ensure_ascii=False)}")
    return "\n".join(parts)

def _extract_json(s: str) -> Dict[str,Any]:
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    first = s.find("{"); last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        cand = s[first:last+1]
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    raise ValueError("Model output did not contain valid JSON.")

# ---------- Model adapters ----------
def _call_model_ollama(messages: List[Dict[str,str]]) -> str:
    url = OLLAMA_HOST.rstrip("/") + "/api/chat"
    body = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return raw.get("message",{}).get("content","")

def _call_model_generic(messages: List[Dict[str,str]]) -> str:
    if not MODEL_URL:
        raise RuntimeError("MODEL_URL not set.")
    headers = {"Content-Type":"application/json"}
    if MODEL_AUTH:
        headers["Authorization"] = MODEL_AUTH
    body = json.dumps({"messages": messages, "stream": False}).encode("utf-8")
    req = urllib.request.Request(MODEL_URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    if isinstance(raw, dict):
        if "message" in raw and isinstance(raw["message"], dict):
            return raw["message"].get("content","")
        if "choices" in raw and raw["choices"]:
            return raw["choices"][0].get("message",{}).get("content","")
        if "content" in raw:
            return raw.get("content","")
    raise RuntimeError("Unrecognized generic model response shape.")

def call_model(messages: List[Dict[str,str]]) -> str:
    if BACKEND.upper() == "OLLAMA":
        return _call_model_ollama(messages)
    elif BACKEND.upper() == "GENERIC":
        return _call_model_generic(messages)
    elif BACKEND.upper() == "AUTO":
        try:
            return _call_model_ollama(messages)
        except Exception:
            return _call_model_generic(messages)
    else:
        raise RuntimeError(f"Unknown BACKEND '{BACKEND}' (use OLLAMA | GENERIC | AUTO).")

# ---------- Auto-retry wrapper ----------
def _ask_model_with_retries(history: List[Dict[str,str]], max_retries: int = 2) -> str:
    """
    Call model; if content is empty or non-JSON, inject a strict JSON reminder and retry.
    """
    attempt = 0
    while True:
        content = call_model(history)
        if content and content.strip():
            return content
        if attempt >= max_retries:
            return content
        history.append({"role":"system","content":
            "REMINDER: Respond ONLY with ONE JSON object per the schema. No prose, no markdown, no code fences, no thinking."})
        attempt += 1

# ---------- optional warmup ----------
def _warmup_ping() -> None:
    print("[warmup] pinging OLLAMA...")
    try:
        payload = {"model": OLLAMA_MODEL,
                   "messages":[{"role":"user","content":"ok"}],
                   "stream": False}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(OLLAMA_HOST.rstrip("/") + "/api/chat",
                                     data=data,
                                     headers={"Content-Type":"application/json"},
                                     method="POST")
        t0 = datetime.datetime.now()
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            _ = resp.read()
        t1 = datetime.datetime.now()
        print(f"[warmup] chat responded in {(t1 - t0).total_seconds():.1f}s")
    except Exception as e:
        print(f"[warmup] chat failed: {e}")

# ---------- UI helpers ----------
def _print_header():
    print("== LLM Desktop Controller (Linux) ==")
    print(f"Backend        : {BACKEND}")
    if BACKEND.upper() in ("OLLAMA", "AUTO"):
        print(f"  OLLAMA_HOST  : {OLLAMA_HOST}")
        print(f"  OLLAMA_MODEL : {OLLAMA_MODEL}")
    if BACKEND.upper() in ("GENERIC", "AUTO"):
        if MODEL_URL:
            print(f"  MODEL_URL    : {MODEL_URL}")
        else:
            print("  MODEL_URL    : (not set)")
    print(f"SAFE_MODE      : {SAFE_MODE}   EXPERT_MODE: {EXPERT_MODE}")
    print(f"HTTP Timeout   : {REQUEST_TIMEOUT}s")
    print("Type ':help' for meta-commands. Ctrl+C to exit.")

def _show_tools():
    for name, t in TOOLS.items():
        print(f"- {name}: {t.desc}")
        print(f"  schema: {json.dumps(t.schema, ensure_ascii=False)}")

def _show_config():
    blob = {
        "BACKEND": BACKEND,
        "OLLAMA_HOST": OLLAMA_HOST,
        "OLLAMA_MODEL": OLLAMA_MODEL,
        "MODEL_URL": MODEL_URL,
        "MODEL_AUTH": ("<set>" if MODEL_AUTH else ""),
        "SAFE_MODE": SAFE_MODE,
        "EXPERT_MODE": EXPERT_MODE,
        "COMMAND_ALLOWLIST": COMMAND_ALLOWLIST,
        "REQUEST_TIMEOUT": REQUEST_TIMEOUT,
        "TIMEOUT_SHELL": TIMEOUT_SHELL,
        "TIMEOUT_PY": TIMEOUT_PY,
        "MAX_TEXT_BYTES": MAX_TEXT_BYTES,
        "SANDBOX_DIR": SANDBOX_DIR,
    }
    print(json.dumps(blob, indent=2))

def _handle_meta(cmd: str) -> bool:
    global SAFE_MODE, EXPERT_MODE, REQUEST_TIMEOUT
    if cmd == ":help":
        print(":safe on|off   - toggle confirmations/allowlist (off => EXPERT_MODE)")
        print(":allow         - show allowlist")
        print(":allow add <token> | :allow rm <token>")
        print(":tools         - list tools and schemas")
        print(":config        - show current config")
        print(":timeout <s>   - set HTTP timeout for model calls")
        return True
    if cmd.startswith(":safe"):
        parts = cmd.split()
        if len(parts) != 2 or parts[1] not in ("on","off"):
            print("Usage: :safe on|off")
            return True
        if parts[1] == "on":
            SAFE_MODE = True
            EXPERT_MODE = False
        else:
            SAFE_MODE = False
            EXPERT_MODE = True
        print(f"SAFE_MODE={SAFE_MODE} EXPERT_MODE={EXPERT_MODE}")
        return True
    if cmd == ":allow":
        print("Allowlist:", ", ".join(COMMAND_ALLOWLIST) if COMMAND_ALLOWLIST else "(empty)")
        return True
    if cmd.startswith(":allow add "):
        token = cmd[len(":allow add "):].strip()
        if token and token not in COMMAND_ALLOWLIST:
            COMMAND_ALLOWLIST.append(token)
            print("Added:", token)
        else:
            print("Nothing added.")
        return True
    if cmd.startswith(":allow rm "):
        token = cmd[len(":allow rm "):].strip()
        try:
            COMMAND_ALLOWLIST.remove(token)
            print("Removed:", token)
        except ValueError:
            print("Token not in allowlist.")
        return True
    if cmd == ":tools":
        _show_tools()
        return True
    if cmd == ":config":
        _show_config()
        return True
    if cmd.startswith(":timeout "):
        try:
            val = int(cmd.split()[1])
            REQUEST_TIMEOUT = max(30, val)
            print(f"Set HTTP timeout to {REQUEST_TIMEOUT}s")
        except Exception:
            print("Usage: :timeout <seconds>")
        return True
    return False

# ---------- controller ----------
def run_controller():
    base_system = SYSTEM_PROMPT.replace("{tool_list}", _tool_catalog())
    history: List[Dict[str,str]] = [{"role":"system", "content": base_system}]
    _warmup_ping()
    _print_header()

    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not user:
            continue

        if user.startswith(":"):
            if _handle_meta(user):
                continue

        history.append({"role":"user","content": user})

        try:
            content = _ask_model_with_retries(history, max_retries=2)
        except Exception as e:
            print(f"[Model Error] {e}")
            continue

        try:
            action = _extract_json(content)
        except Exception as e:
            print(f"[Parse Error] {e}")
            if content and content.strip():
                print("Raw:\n" + content)
            else:
                print("Raw: <empty response from model>")
            # Nudge for next round
            history.append({"role":"assistant","content": '{"action":"final","message":"Please respond in JSON only as specified."}'})
            history.append({"role":"system","content":
                "REMINDER: Respond ONLY with ONE JSON object per the schema. No prose, no markdown, no code fences, no thinking."})
            continue

        if not isinstance(action, dict) or "action" not in action:
            print(f"[Bad Action] {action}")
            continue

        if action["action"] == "final":
            msg = action.get("message","")
            print(f"Agent> {msg}")
            history.append({"role":"assistant","content": json.dumps(action, ensure_ascii=False)})
            continue

        if action["action"] == "call_tool":
            tool_name = action.get("tool")
            args = action.get("args",{}) or {}
            reason = action.get("reason","")
            say_before = action.get("say_before")
            say_after  = action.get("say_after")

            if tool_name not in TOOLS:
                print(f"[Tool Error] Unknown tool: {tool_name}")
                history.append({"role":"assistant","content": json.dumps({"action":"final","message":f"Unknown tool {tool_name}."})})
                continue

            if isinstance(say_before, str) and say_before.strip():
                print(f"Agent> {say_before}")

            tool = TOOLS[tool_name]
            print(f"Agent requests tool: {tool_name}  reason: {reason or '-'}")
            try:
                result = tool.call(args)
                tool_out = {"tool": tool_name, "ok": True, "result": result}
            except ToolError as te:
                tool_out = {"tool": tool_name, "ok": False, "error": str(te)}
            except Exception as e:
                tool_out = {"tool": tool_name, "ok": False, "error": f"{type(e).__name__}: {e}"}

            print("[TOOL RESULT]")
            print(json.dumps(tool_out, ensure_ascii=False))
            history.append({"role":"assistant","content": json.dumps(action, ensure_ascii=False)})
            history.append({"role":"user","content": f"TOOL_RESULT:\n{json.dumps(tool_out, ensure_ascii=False)}"})

            if isinstance(say_after, str) and say_after.strip():
                print(f"Agent> {say_after}")

            continue

        print(f"[Unknown action] {action}")
        history.append({"role":"assistant","content": json.dumps({"action":"final","message":"Unknown action."})})

# ---------- entry ----------
if __name__ == "__main__":
    try:
        run_controller()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")
