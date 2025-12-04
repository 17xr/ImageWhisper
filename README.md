# Image Caption Generator

An image captioning model that uses [facebook/dinov3](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m) as the image encoder and leverages [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) for tokenizer and embeddings. Its decoder follows a modern Qwen 3 inspired architecture to generate high-quality captions. The project includes a FastAPI backend for inference and a Streamlit frontend for interactive use.

## 🧮 Requirements

- Python 3.8+
- A GPU is strongly recommended for inference at reasonable speed
- (Optional) Virtual environment to isolate dependencies

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/17xr/ImageWhisper.git
cd ImageWhisper
```

2. Install the Python dependencies:
```bash
pip install --no-cache-dir -r requirements.txt
```

3. (Optional) If you have CUDA/GPU support, ensure the correct PyTorch/CUDA version is installed.

## ▶️ Running the Application

### 1. Start the Backend

In the project root, run:
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
This starts the HTTP API server (using Uvicorn) on port 8000.

### 2. Start the Frontend

In a separate terminal window (with the backend running), run:
```bash
cd frontend
streamlit run src/main.py
```
This launches the Streamlit UI in your browser.

### 3. Use the Application

- In the Streamlit UI, upload an image (or select a test image) and click "Generate Caption".
- The frontend sends the image to the backend, which runs the model and returns a caption.
- The caption is displayed in the UI.

## 📁 Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   ├── __init__.py
│   │   └── main.py
│   ├── architecture/
│   │   ├── __init__.py
│   │   └── transformer.py
│   ├── model/
│   │   └── model.pt
│   └── utils/
│       ├── __init__.py
│       └── utils.py
├── frontend/
│   └── src/
│       └── main.py
├── LICENSE
├── README.md
└── requirements.txt
```

## 📄 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
