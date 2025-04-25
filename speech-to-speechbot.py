import torch
import gradio as gr
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from TTS.api import TTS


asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_response_with_context(query, context):
    input_text = f"Context: {context}\n\nQuestion: {query}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

def synthesize_audio(text):
    tts.tts_to_file(text=text, file_path="output.wav")
    return "output.wav"


embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Example knowledge base (you can load from files or a DB)
documents = [
    "Electricity can be saved by turning off unused appliances.",
    "LED bulbs consume less power than incandescent bulbs.",
    "Running washing machines on full load saves electricity.",
    "TinyLlama is a small language model optimized for fast inference.",
    "FLAN-T5 is trained with instruction tuning for better generalization."
]
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

def retrieve_context(query, top_k=2):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return " ".join([documents[i] for i in indices[0]])


def chatbot_pipeline(audio):
    if audio is None:
        return "No audio input received.", None

    start_time = time.time()

   
    query = asr(audio)["text"]
    print(f"[USER]: {query}")

    
    context = retrieve_context(query)
    print(f"[CONTEXT]: {context}")

    
    response = generate_response_with_context(query, context)
    print(f"[BOT]: {response}")

    
    audio_path = synthesize_audio(response)

    duration = time.time() - start_time
    print(f"[INFO] Total time: {duration:.2f} seconds")

    return response, audio_path


with gr.Blocks() as demo:
    gr.Markdown("### 🧠 RAG-Enhanced Voice Assistant")
    audio_input = gr.Audio(label="🎤 Speak here", type="filepath", format="wav")
    response_text = gr.Textbox(label="🤖 Bot Response")
    response_audio = gr.Audio(label="🔊 Bot Voice Output")
    submit_btn = gr.Button("💬 Ask Bot")

    submit_btn.click(fn=chatbot_pipeline, inputs=audio_input, outputs=[response_text, response_audio])

demo.launch()