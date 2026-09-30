import os
import sqlite3
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Adaptive AI Coding Assistant")

# Use persistent directory for Render (/data) or fallback locally
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
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM system_config")
    if cursor.fetchone()[0] == 0:
        default_instruction = (
            "You are an expert AI coding assistant. Provide clean, well-documented, "
            "and efficient code. Always explain your logic clearly."
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

def save_chat_message(role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def get_chat_history() -> List[dict]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_history ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": text} for role, text in rows]

init_db()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is missing!")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# You can use free models on OpenRouter like qwen/qwen-2.5-coder-32b-instruct:free
MODEL_NAME = "qwen/qwen-2.5-coder-32b-instruct:free"

class QueryRequest(BaseModel):
    prompt: str

class BehaviorRequest(BaseModel):
    instruction_change: str

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Adaptive Coding Assistant</title>
        <style>
            body { font-family: sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; background: #0f172a; color: #f8fafc; }
            h1 { text-align: center; color: #38bdf8; }
            .box { background: #1e293b; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }
            textarea { width: 100%; height: 70px; background: #0f172a; color: #fff; border: 1px solid #475569; border-radius: 4px; padding: 0.5rem; box-sizing: border-box; }
            button { background: #0284c7; color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 4px; cursor: pointer; margin-top: 0.5rem; font-weight: bold; }
            .msg { padding: 0.8rem; margin: 0.5rem 0; border-radius: 6px; white-space: pre-wrap; }
            .user { background: #1e3a8a; }
            .assistant { background: #064e3b; }
        </style>
    </head>
    <body>
        <h1>Adaptive AI Coding Assistant</h1>
        
        <div class="box">
            <h3>Active System Behavior</h3>
            <p id="currentInstruction">Loading...</p>
            <textarea id="configInput" placeholder="e.g., Always use TypeScript and output only code block without extra explanations."></textarea>
            <button onclick="updateConfig()">Update Instructions</button>
        </div>

        <div class="box">
            <h3>Persistent Chat Session</h3>
            <div id="chatHistory"></div>
            <hr style="border-color:#334155; margin: 1rem 0;">
            <textarea id="promptInput" placeholder="Ask a question or request code..."></textarea>
            <button onclick="sendPrompt()">Send</button>
            <button onclick="clearChat()" style="background:#dc2626; float:right;">Clear History</button>
        </div>

        <script>
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
                    div.innerText = (msg.role === 'user' ? 'You: ' : 'AI: ') + msg.content;
                    container.appendChild(div);
                });
            }

            async function updateConfig() {
                const change = document.getElementById('configInput').value;
                if (!change) return;
                document.getElementById('currentInstruction').innerText = "Updating...";
                const res = await fetch('/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ instruction_change: change })
                });
                const data = await res.json();
                document.getElementById('currentInstruction').innerText = data.new_system_instruction;
                document.getElementById('configInput').value = '';
            }

            async function sendPrompt() {
                const prompt = document.getElementById('promptInput').value;
                if (!prompt) return;
                document.getElementById('promptInput').value = '';
                await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                fetchHistory();
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
def read_behavior():
    return {"current_instruction": get_system_instruction()}

@app.post("/config")
def update_behavior(request: BehaviorRequest):
    current_instruction = get_system_instruction()
    meta_prompt = (
        f"Current System Instructions:\n\"{current_instruction}\"\n\n"
        f"User request to change behavior:\n\"{request.instruction_change}\"\n\n"
        "Rewrite the system instructions to incorporate this change while keeping "
        "core coding capabilities intact. Return ONLY the new system instruction text."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": meta_prompt}]
        )
        new_instruction = response.choices[0].message.content.strip()
        set_system_instruction(new_instruction)
        return {"new_system_instruction": new_instruction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def read_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_history ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return {"history": [{"role": r, "content": c} for r, c in rows]}

@app.delete("/history")
def delete_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()
    return {"status": "cleared"}

@app.post("/ask")
def ask_code(request: QueryRequest):
    current_instruction = get_system_instruction()
    save_chat_message("user", request.prompt)
    
    messages = [{"role": "system", "content": current_instruction}]
    messages.extend(get_chat_history())
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        ai_reply = response.choices[0].message.content
        save_chat_message("assistant", ai_reply)
        return {"response": ai_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
