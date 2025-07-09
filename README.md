# 🌍 DrishtiX – Where Earth Meets Insight

**DrishtiX** is an intelligent super-resolution web platform that transforms low-resolution satellite images into high-resolution outputs using a dual-path approach. It combines classical image processing and deep learning to maximize detail restoration. The system also evaluates image quality using both full-reference and no-reference metrics, making it ideal for real-world satellite applications where ground-truth data is often missing.

---

## 👥 Team DrishtiX
- **Sandeep Sarkar**  
- **Atyasha Bhattacharyya**  
- **Subhanjan Saha**  
- **Ishaan Karmakar**

---

## 🚀 Project Overview

Due to hardware and bandwidth limitations, satellites often capture multiple slightly-shifted low-resolution (LR) images instead of a single high-resolution (HR) one. Our solution leverages two such LR inputs to reconstruct a high-quality SR image with minimal error and high perceptual fidelity — even when ground-truth HR images are unavailable .

---

## 🔬 Methodology

### 🔁 1. Image Alignment  
- Input: Two slightly shifted LR satellite images  
- Technique: **Enhanced Correlation Coefficient (ECC)** based motion model  
- Goal: Sub-pixel image registration to align both inputs precisely

### 🔄 2. Image Fusion  
- Aligned LR images are fused to combine complementary spatial information.

### 🛠️ 3. Dual-Path Super Resolution

#### 🔹 Classical Path:
- **Upsampling** using bicubic interpolation  
- **Degradation modeling**  
- **Iterative Back-Projection** for refinement

#### 🔸 Deep Learning Path:
- Uses the **SwinIR** model (Swin Transformer-based SR)
- Captures long-range dependencies and local attention

### ⚖️ 4. Weighted Fusion of SR Outputs  
- Combines the outputs of both SR branches to produce a robust final SR image  
- Balances perceptual quality and structural accuracy

---

## 📊 Quality Evaluation Metrics

After SR generation, the image quality is evaluated using:

### ✅ Full-Reference (if GT is available):
- **PSNR** – Peak Signal-to-Noise Ratio  
- **SSIM** – Structural Similarity Index  
- **RMSE** – Root Mean Squared Error  
- **MSME** – Mean Square Matrix Error  

### ✅ No-Reference (Blind Assessment):
- **BRISQUE** – Blind/Referenceless Image Spatial Quality Evaluator  
- **NIQE** – Natural Image Quality Evaluator

---

## 🌐 Streamlit Web App Features

- Upload 2 low-res satellite images
- Automatic image alignment and dual SR generation
- Display of SR outputs and all metric scores
- Option to download the final high-resolution image

---

## ⚙️ Tech Stack

- **Frontend**: Streamlit  
- **Backend Models**: PyTorch (SwinIR), OpenCV, SciPy  
- **Metrics**: scikit-image, sewar, OpenCV quality module  
- **Deployment Ready**: Easy to run locally or host on platforms like Hugging Face or Streamlit Cloud

---

## 🖥️ Running Locally

```bash
git clone https://github.com/<your-username>/DrishtiX.git
cd DrishtiX
pip install -r requirements.txt
streamlit run app.py
