# 🧠 RAG-Enhanced Voice Assistant

This project is a speech-to-speech chatbot powered by:
- 🗣 Whisper ASR for speech recognition
- 🔍 Sentence Transformers + FAISS for context retrieval (RAG)
- 🤖 FLAN-T5 for generating natural language responses
- 🗯 Coqui TTS for text-to-speech synthesis
- 🎛 Gradio for a clean web interface
- ⏳ time taken to process in 3-5 seconds
---

## 🚀 Features

- Convert voice input into text using Whisper ASR
- Retrieve relevant knowledge base documents using FAISS
- Generate responses using FLAN-T5 with instruction tuning
- Synthesize responses back into speech
- Interact through a simple Gradio interface

---

## 📦 Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
