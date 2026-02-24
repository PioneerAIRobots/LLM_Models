# LLaMA 3.2 Vision for Chest X-Ray Interpretation

---

🫁 **What if an AI could read a chest X-ray and generate a clinical report — in seconds?**

That's not science fiction anymore. I just built it.

I've been working on fine-tuning **LLaMA 3.2 Vision** — Meta's latest multimodal large language model — specifically for **chest X-ray interpretation**. The result is a model that can look at a chest X-ray and produce clinically meaningful outputs: classifications, findings descriptions, and full radiology-style reports.

---

**Why this matters:**

There are over **2 billion chest X-rays** performed globally every year. Yet radiologist shortages mean reads are delayed — sometimes by hours in critical cases. In low-resource settings, some X-rays never get formally read at all.

A fine-tuned Vision-Language Model doesn't replace a radiologist. But it can:

📋 Generate a first-pass report for radiologist review
🚨 Flag high-priority findings for faster triage
🌍 Bring diagnostic support to under-resourced hospitals
📚 Serve as a teaching tool for medical students

---

**What I built:**

A complete fine-tuning pipeline for **LLaMA 3.2 Vision** on chest X-ray datasets using the **Unsloth** framework — achieving fast, memory-efficient training through **LoRA/QLoRA** parameter-efficient adaptation.

The model supports three task modes:
🔹 **Classification** — Normal / Pneumonia / Pleural Effusion / etc.
🔹 **Captioning** — Generate a descriptive findings summary
🔹 **Visual QA** — Answer clinical questions about an X-ray

Example outputs from fine-tuned inference:
✅ *"Normal chest X-ray. No acute cardiopulmonary findings."*
⚠️ *"Findings suggest right lower lobe pneumonia. Recommend clinical correlation."*
🔴 *"Large left pleural effusion noted. Urgent evaluation advised."*

---

**Tech stack:**
🦙 **LLaMA 3.2 Vision** — Meta's multimodal foundation model
⚡ **Unsloth** — 2× faster fine-tuning, 60% less VRAM
🤗 **Hugging Face Transformers + PEFT** — LoRA/QLoRA adaptation
🔥 **PyTorch + Accelerate** — distributed training support
📊 **ChestX-ray14 / MIMIC-CXR** — open medical imaging datasets

---

**What makes this technically significant:**

Fine-tuning a Vision-Language Model for medical imaging is non-trivial. Medical images require the model to understand both **visual pathology patterns** AND **clinical language** simultaneously. LoRA allows us to adapt an 11B parameter model on a single GPU — making this accessible to researchers without massive compute budgets.

This is part of a broader push toward **Foundation Models for Medical Imaging** — general-purpose models pre-trained at scale, then efficiently adapted for specific clinical tasks.

---

🔗 Full notebook and code on GitHub — link in comments.

If you're working in medical AI, radiology informatics, or multimodal LLMs — I'd love to connect and discuss where this technology is headed.

#MedicalAI #LLM #LLaMA #VisionLanguageModel #ChestXray #Radiology #HealthcareAI #MultimodalAI #DeepLearning #FoundationModels #LoRA #QLoRA #Unsloth #AIinHealthcare #MedicalImaging #NLP #ComputerVision #HuggingFace #GenerativeAI #ClinicalAI

---
---

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

<p align="center">Built with ❤️ for the future of medical AI · <b>Mansoor Ahmad</b> · AI & Robotics Engineer · NSTP Islamabad</p>
```
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





