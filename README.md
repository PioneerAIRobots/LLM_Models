# 🐙 GITHUB README — LLaMA 3.2 Vision Chest X-Ray Fine-Tuning

```markdown
# 🫁 Towards a Foundation Model for Chest X-Ray Interpretation

### LLaMA 3.2 Vision Fine-Tuning on Medical Imaging with Unsloth

<p align="center">
  <img src="https://img.shields.io/badge/LLaMA-3.2%20Vision-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Unsloth-2x%20Faster-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LoRA%2FQLoRA-PEFT-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Task-Medical%20Imaging-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <b>A complete fine-tuning pipeline for adapting LLaMA 3.2 Vision to chest X-ray classification,
  captioning, and report generation — using LoRA/QLoRA on a single GPU.</b>
</p>

---

<img width="1536" height="1024" alt="ChatGPT Image Sep 27, 2025, 02_37_15 PM" src="https://github.com/user-attachments/assets/6798a884-f838-43d0-8c6d-dd9c365eee7d" />


## 🎯 Overview

This repository demonstrates how to fine-tune **Meta's LLaMA 3.2 Vision** on chest X-ray datasets for three clinical tasks:

- 🔍 **Classification** — Identify pathologies (Pneumonia, Effusion, Atelectasis, Normal, etc.)
- 📝 **Report Generation** — Produce radiology-style findings descriptions
- 💬 **Visual Question Answering (VQA)** — Answer clinical questions about X-ray findings

Using **Unsloth**, training is **2× faster** with **60% less VRAM** than standard fine-tuning —
making this accessible on a single consumer or research GPU.

---

## 🌍 Motivation

> There are over **2 billion chest X-rays** performed globally every year.

Radiologist shortages create dangerous delays — particularly in low-resource settings.
Vision-Language Models fine-tuned on medical imaging data offer a path toward:

- Automated first-pass report generation for radiologist review
- High-priority finding flagging for faster clinical triage
- Diagnostic support in under-resourced healthcare systems
- Medical education and training assistance

**This project is a research demonstration — not a clinical product.**

---

## ✨ Features

- ✅ **LLaMA 3.2 Vision fine-tuning** with Unsloth for efficient multimodal training
- ✅ **LoRA / QLoRA** — adapt an 11B model on a single GPU
- ✅ **Medical dataset integration** — ChestX-ray14, MIMIC-CXR, or custom datasets
- ✅ **Three task modes** — classification, captioning, visual QA
- ✅ **Evaluation metrics** — Accuracy, BLEU, ROUGE
- ✅ **End-to-end inference** — raw X-ray image to clinical text output

---

## 🧠 Pipeline

```
Chest X-Ray Image
      ↓
LLaMA 3.2 Vision Encoder (frozen)
      ↓
Cross-Modal Attention — Image + Text
      ↓
LoRA-adapted Language Model Head
      ↓
Clinical Text Output
  → "No acute cardiopulmonary findings."
  → "Right lower lobe pneumonia. Clinical correlation recommended."
  → "Large left pleural effusion. Urgent evaluation advised."
```

**LoRA Config:**
- Rank r=16 | Target: q_proj, v_proj, k_proj, o_proj
- 4-bit QLoRA quantization
- ~1–2% trainable parameters of total model

---

## 📊 Example Outputs

| X-Ray Finding | Model Output |
|---------------|-------------|
| Normal PA film | ✅ "Normal chest X-ray. No acute findings identified." |
| Lower lobe opacity | ⚠️ "Right lower lobe consolidation consistent with pneumonia." |
| Left fluid collection | 🔴 "Large left pleural effusion with compressive atelectasis." |
| Enlarged heart | ⚠️ "Increased cardiac silhouette suggestive of cardiomegaly." |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/your-username/llm-chest-xray.git
cd llm-chest-xray

# 2. Install
pip install unsloth transformers datasets accelerate peft bitsandbytes

# 3. Open notebook
jupyter notebook Llama_3_2_Vision_Finetuning_Unsloth_Xrays.ipynb
```

**Dataset format:**
```python
{
  "image": "path/to/xray.jpg",
  "label": "Pneumonia",
  "report": "Findings suggest right lower lobe consolidation..."
}
```

Supported datasets: [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) · [MIMIC-CXR](https://physionet.org/content/mimic-cxr/)

---

## 📁 Project Structure

```
llm-chest-xray/
├── Llama_3_2_Vision_Finetuning_Unsloth_Xrays.ipynb   # Main notebook
├── data/
│   ├── train/                                          # Training images
│   ├── val/                                            # Validation images
│   └── dataset.json                                    # Labels / reports
├── outputs/
│   ├── checkpoint-*/                                   # LoRA checkpoints
│   └── logs/
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | LLaMA 3.2 Vision (Meta) |
| Fine-tuning | Unsloth — 2× faster, 60% less VRAM |
| Adaptation | PEFT / LoRA / QLoRA (Hugging Face) |
| Framework | PyTorch + Accelerate |
| Pipelines | Hugging Face Transformers |
| Datasets | ChestX-ray14, MIMIC-CXR |

---

## 🔮 Roadmap

| Version | Feature |
|---------|---------|
| v1.0 | Fine-tuning notebook — classification + captioning ✅ |
| v1.1 | Full MIMIC-CXR report generation pipeline |
| v1.2 | Multi-label pathology classification |
| v2.0 | Flask web app — upload X-ray, get AI report |
| v2.1 | DICOM (.dcm) file support |
| v2.2 | GradCAM visual explanation overlays |
| v3.0 | Benchmark: LLaMA vs BioViL vs CheXagent vs MedPaLM |
| v3.1 | RLHF with radiologist feedback |

---

## ⚠️ Disclaimer

This project is **strictly for research and educational purposes**.
It is **not validated for clinical use** and must **not** be used to make or influence
medical decisions. All outputs require review by a qualified radiologist or physician.

---

## 📖 Acknowledgements

- [UnslothAI](https://github.com/unslothai/unsloth) — efficient fine-tuning framework
- [Hugging Face](https://huggingface.co) — Transformers, PEFT, Datasets
- [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) — ChestX-ray14
- [PhysioNet / MIT](https://physionet.org/content/mimic-cxr/) — MIMIC-CXR
- [Meta AI](https://ai.meta.com/llama/) — LLaMA 3.2 Vision

---

## 🤝 Open to Collaboration

Looking to connect with:
- 🏥 Radiologists interested in AI-assisted reporting
- 🧬 Medical AI researchers working on foundation models
- 🤗 VLM researchers pushing multimodal medical AI
- 🚀 Healthcare startups building clinical AI products
- 🌍 Global health technologists expanding diagnostic access

**Let's build medical AI that actually helps people.**

---

## 📜 License

MIT License — open for research and educational use with attribution.

---

⭐ Star · 🍴 Fork · 💬 Contribute · 📤 Share

```






