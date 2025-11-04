# core/models.py
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")


@st.cache_resource
def get_embedder():
    # lightweight, CPU-safe embedder
    return SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
