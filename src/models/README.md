# Models

This folder contains all unimodal models (visual, audio, context) and the fusion network.

## VisualNet (`visual_net.py`)
- A simple CNN that extracts embeddings from video frames.
- Input: image tensor `(batch, 3, H, W)`
- Output: feature vector `(batch, feature_dim)`

## AudioNet (`audio_net.py`)
- A CNN-based model that extracts embeddings from audio features (MFCCs or spectrograms).
- Input: tensor `(batch, 1, time, freq)`
- Output: feature vector `(batch, feature_dim)`

### Notes
- Uses 3 convolutional layers with batch normalization.
- Ends with adaptive pooling and a fully connected layer.
- Designed to be lightweight for local training.


## ContextNet (`context_net.py`)

### Current (Baseline)
- **Embedding + LSTM**: Encodes transcript tokens into a feature vector.
- Input: `(batch, seq_len)` token IDs
- Output: `(batch, feature_dim)`

### Future Extension
- **Transformer Encoder**:
  - Replace LSTM with a 2-layer Transformer encoder.
  - Enable by toggling `self.use_transformer = True`.
  - Allows modeling long-range dependencies in transcripts.

## FusionNet (`fusion_net.py`)
- **Type**: Late Fusion MLP  
- **Inputs**:
  - Visual features (default: 2048-dim from CNN features, e.g. ResNet)  
  - Audio features (default: 40-dim MFCCs or mel-spectrograms)  
  - Context features (optional, e.g. 300-dim embeddings from transcript/text models)  
- **Architecture**:
  - Project each modality into a lower dimension
  - Concatenate projected embeddings
  - Pass through an MLP
  - Output: single logit for binary classification (real vs fake)

**Baseline Flow**
