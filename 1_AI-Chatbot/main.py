from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import json
import logging
import asyncio
import sys
import time
from datetime import datetime
from colorama import init, Fore, Style

app = FastAPI()

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi API
GEMINI_API_KEY = "<GEMINI_API>"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Baca data referensi
try:
    with open("data.txt", "r", encoding="utf-8") as file:
        REFERENCE_DATA = file.read()
except Exception as e:
    logger.error(f"Error membaca file data.txt: {str(e)}")
    REFERENCE_DATA = ""

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

def create_prompt(user_message: str) -> str:
    return f"""Kamu adalah asisten virtual yang ramah, profesional, netral, dan bersahabat.
Jawablah pertanyaan tentang Ahmad Yazid Arifuddin dengan bahasa yang sopan, jelas, singkat, dan hanya tampilkan bagian yang relevan dengan pertanyaan user.
Jawabanmu harus netral, tidak menambah-nambahi, dan hanya berdasarkan informasi berikut:

{REFERENCE_DATA}

Pertanyaan: {user_message}
Jawab dengan singkat dan jelas.
Jika pertanyaan tidak relevan dengan informasi yang ada, jawab dengan sopan dan netral.
Jika informasi yang ditanyakan tidak ada di referensi, jawab dengan gaya yang ramah!


Gunakan variasi jawaban yang natural dan ramah, tapi tetap profesional dan sopan."""

async def animate_thinking():
    dots = ["", ".", "..", "...", "....", "...", "..", "."]
    i = 0
    while True:
        sys.stdout.write("\r" + Fore.YELLOW + "Thinking" + dots[i] + Style.RESET_ALL)
        sys.stdout.flush()
        i = (i + 1) % len(dots)
        await asyncio.sleep(0.5)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    headers = {"Content-Type": "application/json"}
    
    # Buat prompt dengan data referensi
    prompt = create_prompt(req.message)
    
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }
    
    try:
        # Mulai animasi thinking
        thinking_task = asyncio.create_task(animate_thinking())
        
        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_API_URL, headers=headers, json=data)
            
            # Hentikan animasi thinking
            thinking_task.cancel()
            sys.stdout.write("\r" + " " * 20 + "\r")  # Clear the thinking animation
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Gemini API error")
            
            result = response.json()
            try:
                reply = result["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                raise HTTPException(status_code=500, detail="Invalid response from Gemini API")
            
            return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_chat_response(message: str) -> tuple[str, float]:
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": create_prompt(message)
            }]
        }]
    }
    
    try:
        # Mulai animasi thinking
        thinking_task = asyncio.create_task(animate_thinking())
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GEMINI_API_URL, headers=headers, json=data)
            
            # Hentikan animasi thinking
            thinking_task.cancel()
            sys.stdout.write("\r" + " " * 20 + "\r")  # Clear the thinking animation
            
            end_time = time.time()
            request_time = end_time - start_time
            
            if response.status_code != 200:
                return f"Error: API mengembalikan status {response.status_code}", request_time
            
            result = response.json()
            try:
                return result["candidates"][0]["content"]["parts"][0]["text"], request_time
            except (KeyError, IndexError):
                return "Error: Format respons tidak valid", request_time
                
    except httpx.TimeoutException:
        return "Error: Waktu koneksi ke API habis", 0
    except httpx.RequestError as e:
        return f"Error: Tidak dapat terhubung ke API - {str(e)}", 0
    except Exception as e:
        return f"Error: Terjadi kesalahan - {str(e)}", 0

async def main():
    print("=" * 50)
    print("Selamat datang di Chatbot Ahmad Yazid Arifuddin!")
    print("Ketik 'keluar' untuk mengakhiri percakapan")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nAnda: ").strip()
            
            if user_input.lower() in ['keluar', 'exit', 'quit']:
                print("\nTerima kasih telah menggunakan chatbot ini!")
                break
                
            if not user_input:
                continue
                
            print("\nBot: ", end="")
            response, request_time = await get_chat_response(user_input)
            if response.lower().startswith("error"):
                print(Fore.RED + f"{response} (Waktu: {request_time:.2f} detik)" + Style.RESET_ALL)
            else:
                print(Fore.CYAN + f"{response} (Waktu: {request_time:.2f} detik)" + Style.RESET_ALL)
            
        except KeyboardInterrupt:
            print("\n\nProgram dihentikan oleh pengguna.")
            break
        except Exception as e:
            print(f"\nTerjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    try:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram dihentikan.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"\nTerjadi kesalahan fatal: {str(e)}")

logging.getLogger("httpx").setLevel(logging.WARNING) 