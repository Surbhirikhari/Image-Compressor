import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import io

st.set_page_config(page_title="Image Compressor", layout="centered")

st.title("🎨 Image Compressor using K-Means Clustering")

# Upload image
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Choose number of colors
k = st.slider("Select number of colors for compression (K)", min_value=2, max_value=64, value=16)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((200, 200))
    img_np = np.array(img)

    # Flatten image array
    pixels = img_np.reshape(-1, 3)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    # Create compressed image
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
    compressed_img = compressed_pixels.reshape(img_np.shape)

    # Show original and compressed images side by side
    st.subheader("Original vs Compressed Image")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].axis('off')

    ax[1].imshow(compressed_img)
    ax[1].set_title(f"Compressed (k={k})")
    ax[1].axis('off')

    st.pyplot(fig)

    compressed_pil = Image.fromarray(compressed_img)

    # Save to buffer
    import io
    buf = io.BytesIO()
    compressed_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    # Download button
    st.download_button(
        label="📥 Download Compressed Image",
        data=byte_im,
        file_name="compressed_image.jpg",
        mime="image/jpeg"
    )