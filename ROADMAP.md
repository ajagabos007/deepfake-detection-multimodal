# Roadmap - Deepfake Detection (Multimodal)

## 1. Setup & Environment
- [x] Create virtual environment  
- [x] Install dependencies (`numpy`, `opencv`, `librosa`, `torch`, etc.)  
- [x] Setup `requirements.txt` (CPU-only friendly)  

---

## 2. Data Preprocessing
- [x] `video.py` → extract frames, detect faces  
- [x] `audio.py` → extract & preprocess audio (MFCCs etc.)  
- [ ] `context.py` → transcripts (via Whisper, subtitles parsing)  

---

## 3. Feature Extraction
- [x] `extract_visual.py` → convert video frames → embeddings  
- [x] `extract_audio.py` → convert audio → features  
- [ ] `extract_context.py` → convert transcripts → embeddings  

---

## 4. Model Design
- [x] `visual_net.py` → CNN / ResNet for video  
- [x] `audio_net.py` → LSTM / CNN for audio  
- [x] `context_net.py` → Transformer/LSTM for text  
- [x] `fusion_net.py` → late fusion multimodal network (basic draft)  

---

## 5. Training
- [*] `train_visual.py`  
- [*] `train_audio.py`  
- [*] `train_context.py`  
- [*] `train_fusion.py` (skeleton exists, needs full loop)  

---

## 6. Prediction Pipeline
- [ ] `predict.py` → run preprocessing + feature extraction + fusion model → output *real/fake*  

---

## 7. Evaluation
- [ ] Metrics (accuracy, F1, AUC)  
- [ ] Confusion matrix  
- [ ] Compare single-modal vs multimodal performance  

---

## 8. Interface
- [ ] Simple Streamlit/Flask app (`app/main.py`)  
- [ ] Upload video → see prediction  

---

## 9. Docs & Writeup
- [x] Quick Usage & Notes (`README.md`)  
- [x] Roadmap (`ROADMAP.md`)  
- [ ] Detailed report (later, for thesis)  

---

📌 **Current Status:**  
We have preprocessing and feature extraction (except context), plus a basic fusion model.  
Next milestone: **implement missing unimodal models (`visual_net.py`, `audio_net.py`, `context_net.py`)** before full training + prediction.
