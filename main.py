# Importing libraries
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import nest_asyncio
import uvicorn
from langdetect import detect
from deep_translator import GoogleTranslator

# Import custom RAG logic for handling constitutional queries
from RAG_logic import setup_knowledge_base, query_constitution, generate_response

# Handling asyncio issues in interactive environments
nest_asyncio.apply()

# Initialize the FastAPI app
app = FastAPI()

# Allow frontend requests from different origins 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Load the index.html file for the root endpoint; handle case if file is missing
try:
    html = Path("index.html").read_text()
except FileNotFoundError:
    html = "<h1>Error: index.html not found</h1>"

# Loading and embedding the constitution data
@app.on_event("startup")
async def startup_event():
    try:
        setup_knowledge_base()  
    except Exception as e:
        print(f"Error during startup: {e}")

# Root endpoint to serve the index.html page
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=html)

# POST endpoint to handle user questions about the Kenyan Constitution
@app.post("/ask")
async def ask_question(request: Request):
    try:
        # Parse incoming JSON request for question and optional language
        data = await request.json()
        question = data.get("question", "")
        language = data.get("language", None)

        # Validate that a question is provided
        if not question:
            raise HTTPException(status_code=400, detail="Please provide a question.")

        # Detect the language of the input question (e.g., English or Swahili)
        input_lang = detect(question)

        # Translate the question to English if it's in Swahili for processing
        translated_question = question
        if input_lang == "sw":
            translated_question = GoogleTranslator(source='sw', target='en').translate(question)

        # Query the Constitution knowledge base for relevant text chunks
        relevant_chunks = query_constitution(translated_question)
        context = "\n".join(relevant_chunks)
        # Generate a response in English based on the query and context
        answer_en = generate_response(translated_question, context)

        # Translate the answer to Swahili if the input was in Swahili or the target language is Swahili
        if language == "sw" or input_lang == "sw":
            answer = GoogleTranslator(source='en', target='sw').translate(answer_en)
        else:
            answer = answer_en

        # Return the answer in JSON format
        return {"answer": answer}
    except Exception as e:
        # Handle any errors during processing and return a 500 error
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Run the FastAPI app using Uvicorn server when the script is executed directly
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=port, reload=True)
    server = uvicorn.Server(uvicorn_config)
    await server.serve()