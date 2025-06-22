import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# 🔧 Custom Styling
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
    .stButton>button {
        background-color: #ef4444;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Generator for Conditional GAN
class Generator(nn.Module):
    def __init__(self, z_dim=100, label_dim=10, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + label_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        return self.net(x)

# 🔄 One-hot encoding for digit labels
def one_hot(label, num_classes=10):
    vec = torch.zeros(num_classes)
    vec[label] = 1
    return vec.unsqueeze(0)

# 📦 Load trained conditional generator
device = torch.device("cpu")
G = Generator().to(device)
G.load_state_dict(torch.load("cgan_generator.pth", map_location=device))
G.eval()

# 🎨 UI
st.markdown('<div class="title">🧠 Handwritten Digit Image Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Generate MNIST-like digits using your trained Conditional GAN model</div>', unsafe_allow_html=True)

digit = st.selectbox("👉 Choose a digit (0–9):", list(range(10)))

if st.button("🚀 Generate Images"):
    st.markdown(f"### 🖼️ Generated images of digit **{digit}**")
    cols = st.columns(5)

    for i in range(5):
        z = torch.randn(1, 100)
        label = one_hot(digit)
        with torch.no_grad():
            img = G(z, label).view(28, 28).cpu().numpy()
        img = (img + 1) / 2  # Rescale from [-1, 1] to [0, 1]
        cols[i].image(img, width=100, caption=f"Sample {i+1}")
