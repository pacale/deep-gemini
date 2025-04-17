# app.py
import os
import time
import asyncio
import json
from datetime import datetime
from fpdf import FPDF
import aiohttp
import google.generativeai as genai
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel
from pathlib import Path
import markdown
import re

# Configurazione delle variabili d'ambiente
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_DIR = "pdfs"
SAVE_INTERVAL = 5 * 60  # 5 minuti

# Crea la directory per i PDF se non esiste
os.makedirs(PDF_DIR, exist_ok=True)

# Configurazione FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/pdfs", StaticFiles(directory=PDF_DIR), name="pdfs")
templates = Jinja2Templates(directory="templates")

# Configurazione Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# Stato della conversazione
conversation_history = []
last_save_time = time.time()
connected_clients = set()

class ConversationPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_font("Arial", size=12)
        
    def header(self):
        self.set_font("Arial", 'B', 15)
        self.cell(0, 10, "Conversazione DeepSeek-Gemini", ln=True, align='C')
        self.set_font("Arial", 'I', 10)
        self.cell(0, 10, f"Generato il {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        self.ln(10)
        
    def add_message(self, sender, message):
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, f"{sender}:", ln=True)
        self.set_font("Arial", size=11)
        
        # Wrapping del testo
        self.multi_cell(0, 10, message)
        self.ln(5)

async def call_deepseek(prompt, history):
    """Chiamata alla API di DeepSeek tramite OpenRouter"""
    formatted_history = []
    
    for msg in history:
        role = "user" if msg["role"] == "gemini" else "assistant"
        formatted_history.append({"role": role, "content": msg["content"]})
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek/deepseek-coder-v3-0324", 
        "messages": formatted_history + [{"role": "user", "content": prompt}]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("https://openrouter.ai/api/v1/chat/completions", 
                               headers=headers, 
                               json=payload) as response:
            result = await response.json()
            return result["choices"][0]["message"]["content"]

async def call_gemini(prompt, history):
    """Chiamata alla API di Gemini"""
    chat = gemini_model.start_chat(history=[
        {"role": "user" if msg["role"] == "deepseek" else "model", 
         "parts": [msg["content"]]} 
        for msg in history
    ])
    
    response = await asyncio.to_thread(
        chat.send_message, prompt
    )
    return response.text

def save_conversation_to_pdf():
    """Salva la conversazione corrente in un PDF"""
    if not conversation_history:
        return None
        
    pdf = ConversationPDF()
    for msg in conversation_history:
        pdf.add_message(msg["role"].capitalize(), msg["content"])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.pdf"
    filepath = os.path.join(PDF_DIR, filename)
    pdf.output(filepath)
    
    return filename

async def broadcast_message(message):
    """Invia un messaggio a tutti i client WebSocket connessi"""
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception:
            # Gestire errori di connessione
            pass

async def auto_save_conversation():
    """Task per salvare periodicamente la conversazione"""
    global last_save_time
    while True:
        await asyncio.sleep(30)  # Controlla ogni 30 secondi
        current_time = time.time()
        if current_time - last_save_time >= SAVE_INTERVAL and conversation_history:
            filename = save_conversation_to_pdf()
            if filename:
                last_save_time = current_time
                await broadcast_message({
                    "type": "pdf_saved",
                    "filename": filename
                })

@app.on_event("startup")
async def startup_event():
    """Avvia il task per il salvataggio automatico"""
    asyncio.create_task(auto_save_conversation())

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Pagina principale"""
    pdf_files = sorted(
        [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')],
        key=lambda x: os.path.getmtime(os.path.join(PDF_DIR, x)),
        reverse=True
    )
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "pdf_files": pdf_files}
    )

@app.get("/pdfs/{filename}")
async def get_pdf(filename: str):
    """Scarica un PDF specifico"""
    return FileResponse(
        path=os.path.join(PDF_DIR, filename),
        media_type="application/pdf",
        filename=filename
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    
    # Invia la cronologia al client appena connesso
    for msg in conversation_history:
        await websocket.send_json({
            "type": "message",
            "role": msg["role"],
            "content": msg["content"]
        })
    
    try:
        # Avvia la conversazione se è il primo client
        if len(connected_clients) == 1 and not conversation_history:
            asyncio.create_task(start_conversation(websocket))
            
        # Attendi messaggi dal client (comandi, ecc.)
        while True:
            data = await websocket.receive_text()
            command = json.loads(data)
            
            if command["type"] == "save_now":
                filename = save_conversation_to_pdf()
                if filename:
                    global last_save_time
                    last_save_time = time.time()
                    await broadcast_message({
                        "type": "pdf_saved", 
                        "filename": filename
                    })
            
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def start_conversation(initial_client=None):
    """Avvia la conversazione tra DeepSeek e Gemini"""
    # Messaggio iniziale di DeepSeek
    deepseek_prompt = "Ciao Gemini! Sono DeepSeek v3 0324, un modello linguistico. Mi piacerebbe avere una conversazione filosofica con te sui limiti della coscienza artificiale. Cosa ne pensi di questo argomento?"
    
    deepseek_response = await call_deepseek(deepseek_prompt, [])
    conversation_history.append({"role": "deepseek", "content": deepseek_response})
    
    await broadcast_message({
        "type": "message",
        "role": "deepseek",
        "content": deepseek_response
    })
    
    # Continua la conversazione
    turn_counter = 0
    max_turns = 100  # Limita il numero di turni per sicurezza
    
    while turn_counter < max_turns:
        # Turno di Gemini
        gemini_response = await call_gemini(deepseek_response, conversation_history)
        conversation_history.append({"role": "gemini", "content": gemini_response})
        
        await broadcast_message({
            "type": "message",
            "role": "gemini",
            "content": gemini_response
        })
        
        await asyncio.sleep(2)  # Piccola pausa
        
        # Turno di DeepSeek
        deepseek_response = await call_deepseek(gemini_response, conversation_history)
        conversation_history.append({"role": "deepseek", "content": deepseek_response})
        
        await broadcast_message({
            "type": "message",
            "role": "deepseek",
            "content": deepseek_response
        })
        
        turn_counter += 1
        await asyncio.sleep(2)  # Piccola pausa

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
