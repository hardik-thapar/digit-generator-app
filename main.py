import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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

# Streamlit UI
st.title("Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))
if st.button("Generate Images"):
    st.subheader(f"Generated images of digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        z = torch.randn(1, 64)
        with torch.no_grad():
            img = G(z).view(28, 28).numpy()
        img = (img + 1) / 2  # Normalize to [0,1]
        cols[i].image(img, width=100, caption=f"Sample {i+1}")
