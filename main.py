import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import base64

# Add custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #111827;
        color: white;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding-bottom: 20px;
    }
    .subtitle {
        font-size: 18px;
        color: #cccccc;
        text-align: center;
        padding-bottom: 10px;
    }
    .css-1aumxhk {
        background-color: #1f2937 !important;
    }
    .stButton>button {
        background-color: #ef4444;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
    }
    </style>
""", unsafe_allow_html=True)

# Generator model
class Generator(nn.Module):
    def __init__(self, input_size=64, hidden_size=256, output_size=784):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# Load model
G = Generator().cpu()
G.load_state_dict(torch.load("mnist_generator.pth", map_location=torch.device('cpu')))
G.eval()

# UI
st.markdown('<div class="title">🧠 Handwritten Digit Image Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Generate MNIST-like digits using your own trained GAN model</div>', unsafe_allow_html=True)

digit = st.selectbox("👉 Choose a digit (0–9):", list(range(10)))

if st.button("🚀 Generate Images"):
    st.markdown(f"### 🖼️ Generated images of digit **{digit}**")

    cols = st.columns(5)
    for i in range(5):
        z = torch.randn(1, 64)
        with torch.no_grad():
            img = G(z).view(28, 28).numpy()
        img = (img + 1) / 2  # Normalize to [0,1]
        cols[i].image(img, width=100, caption=f"Sample {i+1}")
