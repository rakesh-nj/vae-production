import streamlit as st
import numpy as np
from main import generate_image_from_latent,manipulate_latent,vae
import torch

# Title and Sidebar
st.title("Mini project - Image Generation and Style transfer using with Variational AutoEncoders")
st.sidebar.header("Adjust Attributes")

# Latent variable sliders
adjustments = {}
for i in range(5):  # Assume 5 latent dimensions to adjust
    adjustments[i] = st.sidebar.slider(f"Latent Dimension {i+1}", -3.0, 3.0, 0.0)

# Button to generate new image
if st.button("Generate Image"):
    # Generate latent vector
    latent_vector = torch.randn(1, 128).to('cpu')
    adjusted_latent = manipulate_latent(latent_vector, adjustments)
    generated_image = generate_image_from_latent(vae, adjusted_latent)
    
    # Display image
    st.image(generated_image, caption="Generated Image", use_container_width=True)

