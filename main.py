import os
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Adaptive AI Coding Assistant")

# Database setup for persistent behavior storage
DB_FILE = "assistant.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_config (
            id INTEGER PRIMARY KEY,
            instruction TEXT NOT NULL
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

# Initialize DB on startup
init_db()

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing!")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"

class QueryRequest(BaseModel):
    prompt: str

class BehaviorRequest(BaseModel):
    instruction_change: str


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves a lightweight Web GUI."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Adaptive Coding Assistant</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; background: #0f172a; color: #f8fafc; }
            h1 { text-align: center; color: #38bdf8; }
            .box { background: #1e293b; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }
            textarea { width: 100%; height: 80px; background: #0f172a; color: #fff; border: 1px solid #475569; border-radius: 4px; padding: 0.5rem; box-sizing: border-box; }
            button { background: #0284c7; color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 4px; cursor: pointer; margin-top: 0.5rem; font-weight: bold; }
            button:hover { background: #0369a1; }
            pre { background: #0f172a; padding: 1rem; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; color: #cbd5e1; }
            .badge { background: #334155; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.85rem; color: #38bdf8; }
        </style>
    </head>
    <body>
        <h1>Adaptive AI Coding Assistant</h1>
        
        <div class="box">
            <h3>Active System Behavior</h3>
            <p id="currentInstruction">Loading current instructions...</p>
            <h4>Modify Behavior</h4>
            <textarea id="configInput" placeholder="e.g., Always use TypeScript and output only code block without extra explanations."></textarea>
            <button onclick="updateConfig()">Update Instructions</button>
        </div>

        <div class="box">
            <h3>Ask Code Question</h3>
            <textarea id="promptInput" placeholder="e.g., Write an Express.js server route for user authentication."></textarea>
            <button onclick="sendPrompt()">Submit Task</button>
            <h4>Response:</h4>
            <pre id="output">Output will appear here...</pre>
        </div>

        <script>
            async function fetchConfig() {
                const res = await fetch('/config');
                const data = await res.json();
                document.getElementById('currentInstruction').innerText = data.current_instruction;
            }

            async function updateConfig() {
                const change = document.getElementById('configInput').value;
                if (!change) return;
                document.getElementById('currentInstruction').innerText = "Updating rules...";
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
                document.getElementById('output').innerText = "Generating code...";
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                const data = await res.json();
                document.getElementById('output').innerText = data.response || data.detail;
            }

            fetchConfig();
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
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=meta_prompt
        )
        new_instruction = response.text.strip()
        set_system_instruction(new_instruction)
        return {
            "message": "System prompt updated successfully",
            "new_system_instruction": new_instruction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_code(request: QueryRequest):
    current_instruction = get_system_instruction()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=request.prompt,
            config=types.GenerateContentConfig(
                system_instruction=current_instruction
            )
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
