import streamlit as st
import time
from RAG_logic  import *


# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="BOBA", page_icon="🇰🇪", layout="centered")

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
        body { background-color: #121212; color: white; }
        .flag-stripe { height: 8px; width: 100%; }
        .black-stripe { background-color: black; }
        .red-stripe { background-color: red; }
        .white-stripe { background-color: white; }
        .green-stripe { background-color: green; }
        .header {
            display: flex; align-items: center; justify-content: center;
            gap: 15px; margin: 20px 0;
        }
        .header img { height: 50px; }
        .header h1 { color: #ff0000; font-size: 32px; margin: 0; }
        div[data-testid="stTextArea"] textarea {
            background-color: #2a2a2a; color: white;
            border: 2px solid white; border-radius: 5px;
        }
        .response-box {
            background: #1e1e1e; padding: 15px; border-left: 6px solid red;
            border-radius: 5px; min-height: 100px;
            white-space: pre-wrap; font-family: Consolas, monospace;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- FLAG STRIPES (TOP) ----------------------
st.markdown("""
<div class="flag-stripe black-stripe"></div>
<div class="flag-stripe white-stripe"></div>
<div class="flag-stripe red-stripe"></div>
<div class="flag-stripe white-stripe"></div>
<div class="flag-stripe green-stripe"></div>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("""
<div class="header">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/1200px-Flag_of_Kenya.svg.png" />
    <h1>Kenyan Constitution Chatbot</h1>
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/1200px-Flag_of_Kenya.svg.png" />
</div>
""", unsafe_allow_html=True)

# ---------------------- INPUT CONTROLS ----------------------
language = st.selectbox("Language:", ["English", "Swahili"])
question = st.text_area("Ask about Kenya's Constitution:")

# ---------------------- ASK BUTTON ----------------------
if st.button("Ask", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        placeholder = st.empty()
        placeholder.markdown('<div class="response-box">Thinking...</div>', unsafe_allow_html=True)

        # Call your chatbot logic directly (no HTTP call)
        answer = chatbot_response(question, "sw" if language == "Swahili" else "en")

        # Typewriter animation
        typed_text = ""
        for char in answer:
            typed_text += char
            placeholder.markdown(f'<div class="response-box">{typed_text}</div>', unsafe_allow_html=True)
            time.sleep(0.015)

# ---------------------- FLAG STRIPES (BOTTOM) ----------------------
st.markdown("""
<div class="flag-stripe black-stripe"></div>
<div class="flag-stripe white-stripe"></div>
<div class="flag-stripe red-stripe"></div>
<div class="flag-stripe white-stripe"></div>
<div class="flag-stripe green-stripe"></div>
""", unsafe_allow_html=True)
