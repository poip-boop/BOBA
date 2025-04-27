from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import nest_asyncio
import uvicorn
from langdetect import detect
from deep_translator import GoogleTranslator
from RAG_logic import setup_knowledge_base, query_constitution, generate_response

# Apply nest_asyncio to fix asyncio issues in interactive environments
nest_asyncio.apply()

app = FastAPI()

# Add CORS middleware to allow frontend requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load index.html (ensure it's in the project root)
try:
    html = Path("index.html").read_text()
except FileNotFoundError:
    html = "<h1>Error: index.html not found</h1>"

@app.on_event("startup")
async def startup_event():
    try:
        setup_knowledge_base()  # Load and embed Constitution
    except Exception as e:
        print(f"Error during startup: {e}")

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=html)

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")
        language = data.get("language", None)

        if not question:
            raise HTTPException(status_code=400, detail="Please provide a question.")

        # Detect input language if not provided
        input_lang = detect(question)

        # Translate to English if input is Swahili
        translated_question = question
        if input_lang == "sw":
            translated_question = GoogleTranslator(source='sw', target='en').translate(question)

        relevant_chunks = query_constitution(translated_question)
        context = "\n".join(relevant_chunks)
        answer_en = generate_response(translated_question, context)

        # Translate answer to Swahili if input was Swahili or target language is Swahili
        if language == "sw" or input_lang == "sw":
            answer = GoogleTranslator(source='en', target='sw').translate(answer_en)
        else:
            answer = answer_en

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=port, reload=True)
    server = uvicorn.Server(uvicorn_config)
    await server.serve()
