import gradio as gr
from transformers import pipeline
import numpy as np
import os
path = "/Users/vyomesh/AI/expo/signlanguage" 
os.makedirs(path, exist_ok=True)

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    text = transcriber({"sampling_rate": sr, "raw": y})["text"]
    images = [f'/Users/vyomesh/AI/signlanguage/{c}.png' for c in text.lower() if c.isalpha()]
    return gr.Gallery(images, label='images',object_fit='scale-down',min_width=50, rows = 3, columns = 5)

demo =interface = gr.Interface(transcribe, gr.Audio(sources=["microphone"]), "gallery", theme="Taithrah/Minimal",)

demo.launch()
