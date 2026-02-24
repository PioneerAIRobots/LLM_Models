🫁 Towards a Foundation Model for Chest X-Ray Interpretation
Fine-Tuning LLaMA 3.2 Vision on Medical Imaging with Unsloth
<p align="center">
  <img src="https://img.shields.io/badge/LLaMA_3.2-Vision-purple?style=for-the-badge&logo=meta"/>
  <img src="https://img.shields.io/badge/Unsloth-Fast%20Fine--Tuning-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LoRA%2FQLoRA-PEFT-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Task-Medical%20Imaging-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge"/>
</p>
<p align="center">
  <b>Adapting multimodal large language models for chest X-ray classification, report generation, and visual question answering — using parameter-efficient fine-tuning on consumer hardware.</b>
</p>


<img width="1536" height="1024" alt="ChatGPT Image Sep 27, 2025, 02_37_15 PM" src="https://github.com/user-attachments/assets/6798a884-f838-43d0-8c6d-dd9c365eee7d" />



🩺 Overview
Radiology is one of the most data-rich and AI-ready specialties in medicine — yet most deployed AI tools are narrow classifiers that output a label, nothing more. This project explores a different direction: fine-tuning a vision-language foundation model (LLaMA 3.2 Vision) to not only classify chest X-rays but to explain findings in natural language, answer clinical questions, and generate structured radiology reports.
Using Unsloth for memory-efficient training and LoRA/QLoRA for parameter-efficient adaptation, the entire fine-tuning pipeline runs on a single GPU — making this accessible for researchers without access to multi-node clusters.

🎯 What This Project Covers
TaskDescriptionClassificationPredicting pathology labels (Normal, Pneumonia, Effusion, etc.)Report GenerationProducing structured radiology-style findings from an X-rayVisual QA (VQA)Answering free-form clinical questions about an imageImage CaptioningGenerating descriptive summaries of radiographic findings

✨ Key Features

🦙 LLaMA 3.2 Vision — state-of-the-art multimodal LLM with strong visual reasoning
⚡ Unsloth integration — 2x faster training, up to 70% less VRAM vs standard HuggingFace training
🔧 LoRA / QLoRA — train only a fraction of parameters; full fine-tuning quality at a fraction of the cost
🏥 Medical imaging focused — designed specifically around chest X-ray datasets and clinical language
📊 Multi-metric evaluation — Accuracy, BLEU, ROUGE depending on the task
🔄 Configurable dataset pipeline — plug in ChestX-ray14, MIMIC-CXR, or your own dataset


🧠 Why Vision-Language Models for Radiology?
Traditional CNN classifiers answer: "Is this pneumonia? Yes/No."
Vision-Language Models answer: "There is increased opacity in the right lower lobe with blunting of the costophrenic angle, findings consistent with consolidation and possible pleural effusion. Recommend clinical correlation."
That distinction matters enormously in clinical practice. Radiologists don't issue binary labels — they write reports. VLMs move AI closer to that workflow.
Input:  Chest X-ray image + "What abnormalities are present?"
Output: "Findings suggest bilateral lower lobe infiltrates 
         consistent with pneumonia. No pneumothorax detected. 
         Cardiac silhouette within normal limits."

🏗️ System Architecture
Chest X-ray Image
        ↓
Vision Encoder (LLaMA 3.2 Vision Backbone)
        ↓
Visual Feature Extraction
        ↓
Cross-Modal Fusion Layer
        ↓
LLaMA Language Model Head
        ↓
LoRA Adapter (fine-tuned weights)
        ↓
Output: Classification Label / Report Text / VQA Answer

🛠️ Tech Stack
ComponentLibraryPurposeFoundation ModelLLaMA 3.2 VisionMultimodal vision-language baseFast Fine-TuningUnsloth2× speed, 70% VRAM reductionPEFT AdaptationPEFT / LoRA / QLoRAParameter-efficient trainingModel HubHuggingFace TransformersModel loading & tokenizationTraining EnginePyTorch + AccelerateGPU training orchestrationQuantizationBitsAndBytes4-bit/8-bit model quantization

🚀 Quick Start
1. Clone the repository
bashgit clone https://github.com/your-username/llm-chest-xray.git
cd llm-chest-xray
2. Install dependencies
bashpip install unsloth transformers datasets accelerate peft bitsandbytes
3. Prepare your dataset
The notebook supports any dataset formatted as image-text pairs. Recommended datasets:
DatasetImagesLabelsAccessNIH ChestX-ray14112,12014 pathology labelsNIHMIMIC-CXR227,835Radiology reportsPhysioNetCheXpert224,31614 observationsStanford
Dataset format expected:
json{
  "image": "path/to/xray.jpg",
  "text": "Findings: Bilateral lower lobe consolidation consistent with pneumonia."
}
4. Launch the notebook
bashjupyter notebook
Open Llama_3_2_Vision_Finetuning_Unsloth_Xrays.ipynb and follow the cells.

📁 Project Structure
llm-chest-xray/
├── Llama_3_2_Vision_Finetuning_Unsloth_Xrays.ipynb   # Main training notebook
├── data/                                               # X-ray dataset (user-provided)
│   ├── train/
│   ├── val/
│   └── test/
├── outputs/                                            # Saved models, checkpoints, logs
├── requirements.txt
└── README.md

📊 Example Inference Results
Input: Unseen chest X-ray from validation set
Input TypeModel OutputClassification✅ "Normal Chest X-ray"Pathology detected⚠️ "Findings suggest right lower lobe pneumonia"Report generation📋 "The cardiac silhouette is normal in size. There is patchy opacity in the right lower lobe. No pleural effusion. Impression: Community-acquired pneumonia."VQA❓ Q: "Is there cardiomegaly?" → A: "No, cardiac silhouette is within normal limits."
Evaluation metrics on validation set vary by task:

Classification → Accuracy, AUC-ROC
Report Generation → BLEU-4, ROUGE-L, ClinicalBERT similarity
VQA → Exact match, token-level F1


⚙️ LoRA Configuration
pythonmodel = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16,           # LoRA rank
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 42,
)

🔮 Roadmap
VersionPlanned Featurev1.1Multi-label pathology classification headv1.2MIMIC-CXR report generation fine-tuningv1.3ClinicalBERT-based evaluation metricsv2.0Structured report generation (Findings / Impression sections)v2.1Radiology VQA benchmark evaluation (VQA-RAD, PathVQA)v3.0Multi-modal RAG — retrieve similar cases, generate grounded reports

📖 Acknowledgements

UnslothAI — for making LLM fine-tuning accessible on consumer hardware
HuggingFace — Transformers, PEFT, Datasets ecosystem
NIH / PhysioNet — for open medical imaging datasets
Meta AI — LLaMA 3.2 Vision model


⚠️ Disclaimer
This project is for research and educational purposes only. It is not intended for clinical use, medical diagnosis, or patient care. All outputs should be reviewed by qualified medical professionals. Do not use this system to make clinical decisions.

🤝 Open to Collaboration
Interested in collaborating on:

🏥 Clinical validation studies
🔬 Medical imaging research
🧬 Multimodal AI for healthcare
📊 Radiology AI benchmarking

Open an issue or connect on LinkedIn.

⭐ Support
If this project is useful to your research:

⭐ Star the repository
🍴 Fork and build on it
💬 Cite it in your work
📤 Share with the medical AI community



