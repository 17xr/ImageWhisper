import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Image Caption Generator", page_icon="🖼️", layout="centered"
)

st.title("🖼️ Image Caption Generator")
st.caption("#### ✨ Generate smart, AI-powered captions for any image.")

st.divider()

common_image_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg"]
uploaded_file = st.file_uploader(
    label="Upload an image", type=common_image_extensions, label_visibility="collapsed"
)

url = "http://127.0.0.1:8000/caption"

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    uploaded_file.seek(0)
    with st.spinner("⚡ Generating captions..."):
        files = {"file": uploaded_file}
        response = requests.post(url, files=files)
        captions = response.json().get("captions", [])

    st.subheader("📝 Generated Captions")

    for idx, caption in enumerate(captions, start=1):
        with st.expander(f"Caption {idx}"):
            st.markdown(f"{caption.capitalize()}")
