# Spatial-Temporal Attention Network for Depression Recognition from Facial Videos

## Dataset
The dataset used is **AVEC2014**, which is available from the original paper authors upon request.

## Paper Reference
This repository is an implementation of the paper:

**"Spatial-Temporal Attention Network for Depression Recognition from Facial Videos"**  
📎 [Read Paper](https://www.researchgate.net/profile/Yuchen-Pan-7/publication/373658727_Spatial-Temporal_Attention_Network_for_Depression_Recognition_from_Facial_Videos/links/64f6c40c827074313ffda10c/Spatial-Temporal-Attention-Network-for-Depression-Recognition-from-Facial-Videos.pdf)

---

## Repository Overview

### 1. `extract_frames.py`
- Used for extracting **RGB frames** from the 300 videos of AVEC2014.
- These videos belong to **84 participants**.

### 2. `face_detect_avec2014.py`
- Implements **face alignment** using **MediaPipe**.
- Optimized for **parallel processing on CPU**.
- Unlike the paper (which uses **6 RTX Titan GPUs**), this approach saves **every 3rd frame** and performs face alignment for efficiency.

### 3. `spatial_temporal_network.ipynb`
- The **main training and logic** file written in **PyTorch**.
- The paper uses **AVEC2013/2014**, but this implementation uses only **AVEC2014**, so results may slightly differ.
- Training was done for **200 epochs** on a **T4 GPU**:
  - **Training videos:** 200  
  - **Testing videos:** 100  
- The architecture and parameters are kept the same as in the original paper.

---

## Results

| Metric | Paper Results | Our Results |
|--------|---------------|-------------|
| MAE    | 6.00          | 9.36        |
| RMSE   | 7.75          | 11.94       |

- The difference in performance is due to **dataset limitations**.
- However, all other factors are aligned with the original paper.
- We aim to:
  - Integrate **audio modality**.
  - Improve performance further.
  - Hopefully, **beat SOTA (state-of-the-art) models**.

---

## Thanks for Reading!
We hope this work helps others working on **multimodal depression recognition**.
Thats just the start we are looking to make our contribution in this area as much as notable as possible. Thanks!
