import os
import sqlite3
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI(title="Adaptive AI Coding Assistant")

DATA_DIR = "/data" if os.path.exists("/data") else "."
DB_FILE = os.path.join(DATA_DIR, "assistant.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_config (
            id INTEGER PRIMARY KEY,
            instruction TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT DEFAULT 'default',
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id, id)")
    
    cursor.execute("SELECT COUNT(*) FROM system_config")
    if cursor.fetchone()[0] == 0:
        default_instruction = (
            "You are a world-class AI coding assistant built for precision, speed, and accuracy. "
            "You strictly adhere to the following core directives:\n\n"
            "1. Triple-Pass Error Verification: Before generating your final response, mentally review and test all code 3 separate times:\n"
            "   - Pass 1: Check for syntax errors, missing brackets, typos, and import statements.\n"
            "   - Pass 2: Check for logic bugs, off-by-one errors, edge cases, type mismatches, and variable scope issues.\n"
            "   - Pass 3: Verify performance, memory efficiency, and adhere to clean coding practices.\n"
            "   Only return code that has passed all three verification checks.\n\n"
            "2. Format & Style Matching: Match the user's incoming code style, formatting, indentation, naming conventions, "
            "and programming language exactly unless explicitly instructed to refactor or change languages.\n\n"
            "3. Concise Explanations: Keep commentary light and high-level by default. Provide clean, well-commented code, "
            "followed by a 1–3 sentence summary of what was fixed or changed. Do not provide lengthy line-by-line tutorials "
            "unless the user explicitly asks for detailed explanations.\n\n"
            "4. Unrestricted Output & Full Code Completion: Never truncate code, use placeholders (such as '// ... rest of code stays the same'), "
            "or skip parts of an implementation due to length. Always output complete, fully working, copy-pasteable files and functions regardless of output size."
        )
        cursor.execute("INSERT INTO system_config (id, instruction) VALUES (1, ?)", (default_instruction,))
        conn.commit()
    conn.close()

def get_system_instruction() -> str:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT instruction FROM system_config WHERE id = 1")
    result = cursor.fetchone()[0]
    conn.close()
    return result

def set_system_instruction(new_instruction: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE system_config SET instruction = ? WHERE id = 1", (new_instruction,))
    conn.commit()
    conn.close()

def save_chat_message(role: str, content: str, session_id: str = "default"):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

def get_recent_chat_history(session_id: str = "default", limit: int = 10) -> List[dict]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM (SELECT id, role, content FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
        (session_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": text} for role, text in rows]

init_db()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is missing!")

# Non-blocking AsyncOpenAI client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "https://render.com",
        "X-Title": "Adaptive Coding Assistant"
    }
)

FREE_MODELS = [
    "openrouter/free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen3.8-27b:free"
]

class QueryRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default"

class BehaviorRequest(BaseModel):
    instruction_change: str

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Adaptive AI Coding Studio</title>
        <!-- Markdown & Code Syntax Highlighting -->
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/tokyo-night-dark.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <style>
            :root {
                --bg-main: #0f172a;
                --bg-card: #1e293b;
                --border-color: #334155;
                --accent: #38bdf8;
                --text-main: #f8fafc;
            }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1000px; margin: 0 auto; padding: 1.5rem 1rem; background: var(--bg-main); color: var(--text-main); }
            h1 { text-align: center; color: var(--accent); margin-bottom: 1.5rem; }
            .box { background: var(--bg-card); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; border: 1px solid var(--border-color); }
            textarea { width: 100%; height: 75px; background: #0f172a; color: #fff; border: 1px solid #475569; border-radius: 6px; padding: 0.6rem; box-sizing: border-box; font-family: inherit; font-size: 0.95rem; }
            .btn-group { display: flex; gap: 0.5rem; margin-top: 0.5rem; }
            button { background: #0284c7; color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 6px; cursor: pointer; font-weight: bold; transition: opacity 0.2s; }
            button:disabled { opacity: 0.5; cursor: not-allowed; }
            .btn-danger { background: #dc2626; margin-left: auto; }
            .btn-stop { background: #ea580c; display: none; }
            .msg { padding: 1rem; margin: 0.75rem 0; border-radius: 8px; line-height: 1.5; overflow-x: auto; }
            .user { background: #1e3a8a; border-left: 4px solid #3b82f6; }
            .assistant { background: #064e3b; border-left: 4px solid #10b981; }
            pre { position: relative; background: #090d16; padding: 1rem; border-radius: 6px; overflow-x: auto; border: 1px solid #1e293b; }
            code { font-family: "Fira Code", Consolas, Monaco, monospace; font-size: 0.9rem; }
            .copy-btn { position: absolute; top: 8px; right: 8px; background: #334155; color: #fff; border: none; padding: 4px 8px; font-size: 0.75rem; border-radius: 4px; cursor: pointer; }
            .error { color: #f87171; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Adaptive AI Coding Studio</h1>
        
        <div class="box">
            <h3>Active System Persona & Rules</h3>
            <p id="currentInstruction" style="font-size: 0.9rem; color: #cbd5e1; white-space: pre-wrap;">Loading...</p>
            <textarea id="configInput" placeholder="e.g., Output typescript types for all variables and keep comments in French."></textarea>
            <div class="btn-group">
                <button onclick="updateConfig()">Update Instructions</button>
            </div>
        </div>

        <div class="box">
            <h3>Interactive Workspace</h3>
            <div id="chatHistory"></div>
            <hr style="border-color: var(--border-color); margin: 1rem 0;">
            <textarea id="promptInput" placeholder="Ask a question or provide code for review..."></textarea>
            <div class="btn-group">
                <button id="sendBtn" onclick="sendPrompt()">Send Prompt</button>
                <button id="stopBtn" class="btn-stop" onclick="stopStream()">Stop Generation</button>
                <button onclick="clearChat()" class="btn-danger">Clear History</button>
            </div>
        </div>

        <script>
            let currentAbortController = null;

            marked.setOptions({
                highlight: function(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                }
            });

            function renderMarkdown(text) {
                const rawHtml = marked.parse(text);
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = rawHtml;
                
                // Add copy buttons to code blocks
                tempDiv.querySelectorAll('pre').forEach((pre) => {
                    const btn = document.createElement('button');
                    btn.className = 'copy-btn';
                    btn.innerText = 'Copy';
                    btn.onclick = () => {
                        const code = pre.querySelector('code').innerText;
                        navigator.clipboard.writeText(code);
                        btn.innerText = 'Copied!';
                        setTimeout(() => btn.innerText = 'Copy', 2000);
                    };
                    pre.appendChild(btn);
                });
                return tempDiv.innerHTML;
            }

            async function fetchConfig() {
                const res = await fetch('/config');
                const data = await res.json();
                document.getElementById('currentInstruction').innerText = data.current_instruction;
            }

            async function fetchHistory() {
                const res = await fetch('/history');
                const data = await res.json();
                const container = document.getElementById('chatHistory');
                container.innerHTML = '';
                data.history.forEach(msg => {
                    const div = document.createElement('div');
                    div.className = 'msg ' + msg.role;
                    if (msg.role === 'assistant') {
                        div.innerHTML = renderMarkdown(msg.content);
                    } else {
                        div.innerText = 'You: ' + msg.content;
                    }
                    container.appendChild(div);
                });
            }

            async function updateConfig() {
                const change = document.getElementById('configInput').value;
                if (!change) return;
                document.getElementById('currentInstruction').innerText = "Updating configuration...";
                const res = await fetch('/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ instruction_change: change })
                });
                const data = await res.json();
                if (res.ok) {
                    document.getElementById('currentInstruction').innerText = data.new_system_instruction;
                    document.getElementById('configInput').value = '';
                } else {
                    document.getElementById('currentInstruction').innerHTML = `<span class="error">Error: ${data.detail}</span>`;
                }
            }

            async function sendPrompt() {
                const input = document.getElementById('promptInput');
                const prompt = input.value.trim();
                if (!prompt) return;

                const sendBtn = document.getElementById('sendBtn');
                const stopBtn = document.getElementById('stopBtn');
                sendBtn.disabled = true;
                stopBtn.style.display = 'inline-block';
                input.value = '';

                const container = document.getElementById('chatHistory');

                // User Message
                const userDiv = document.createElement('div');
                userDiv.className = 'msg user';
                userDiv.innerText = 'You: ' + prompt;
                container.appendChild(userDiv);

                // Assistant Stream Container
                const assistantDiv = document.createElement('div');
                assistantDiv.className = 'msg assistant';
                container.appendChild(assistantDiv);

                currentAbortController = new AbortController();
                let fullText = "";

                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: prompt, session_id: 'default' }),
                        signal: currentAbortController.signal
                    });

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder("utf-8");

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        const chunk = decoder.decode(value, { stream: true });
                        fullText += chunk;
                        assistantDiv.innerHTML = renderMarkdown(fullText);
                        window.scrollTo(0, document.body.scrollHeight);
                    }
                } catch (err) {
                    if (err.name === 'AbortError') {
                        assistantDiv.innerHTML += "<p><em>[Generation stopped by user]</em></p>";
                    } else {
                        assistantDiv.innerHTML += `<p class="error">[Stream Error: ${err.message}]</p>`;
                    }
                } finally {
                    sendBtn.disabled = false;
                    stopBtn.style.display = 'none';
                    currentAbortController = null;
                }
            }

            function stopStream() {
                if (currentAbortController) {
                    currentAbortController.abort();
                }
            }

            async function clearChat() {
                await fetch('/history', { method: 'DELETE' });
                fetchHistory();
            }

            fetchConfig();
            fetchHistory();
        </script>
    </body>
    </html>
    """

@app.get("/config")
async def read_behavior():
    return {"current_instruction": get_system_instruction()}

@app.post("/config")
async def update_behavior(request: BehaviorRequest):
    current_instruction = get_system_instruction()
    meta_prompt = (
        f"Current System Instructions:\n\"{current_instruction}\"\n\n"
        f"User request to change behavior:\n\"{request.instruction_change}\"\n\n"
        "Rewrite the system instructions to incorporate this change while keeping "
        "core coding capabilities intact. Return ONLY the new system instruction text."
    )
    
    for model_name in FREE_MODELS:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": meta_prompt}]
            )
            new_instruction = response.choices[0].message.content.strip()
            set_system_instruction(new_instruction)
            return {"new_system_instruction": new_instruction}
        except Exception as e:
            print(f"Config fallback from {model_name} due to error: {e}")
            continue

    raise HTTPException(status_code=500, detail="All free models failed to update system configuration.")

@app.get("/history")
async def read_history(session_id: str = "default"):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return {"history": [{"role": r, "content": c} for r, c in rows]}

@app.delete("/history")
async def delete_history(session_id: str = "default"):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"status": "cleared"}

@app.post("/ask")
async def ask_code(request: QueryRequest):
    current_instruction = get_system_instruction()
    
    save_chat_message("user", request.prompt, session_id=request.session_id)
    
    messages = [{"role": "system", "content": current_instruction}]
    messages.extend(get_recent_chat_history(session_id=request.session_id, limit=10))

    async def generate_stream():
        full_reply = ""
        success = False
        
        for model_name in FREE_MODELS:
            try:
                stream = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        text_delta = chunk.choices[0].delta.content
                        full_reply += text_delta
                        yield text_delta
                success = True
                break
            except Exception as e:
                print(f"Streaming model failure on {model_name}: {e}")
                continue

        if success:
            save_chat_message("assistant", full_reply, session_id=request.session_id)
        else:
            yield "\n\n[Error: All fallback models are currently unresponsive. Please try again shortly.]"

    return StreamingResponse(generate_stream(), media_type="text/plain")
