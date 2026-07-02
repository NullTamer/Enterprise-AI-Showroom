# Project 2: Financial Fine-Tuning & Behavioral Syntax Alignment

This module contains the training recipe, configuration artifacts, and performance evaluations for adapting a base large language model (**Llama-3-8B**) to recognize, parse, and respond using professional financial syntax and alternative data structures.

## 📊 Executive Summary
Standard base models frequently hallucinate or generalize financial metrics, losing critical nuance required for alternative asset management and portfolio intelligence. This project executes a targeted **QLoRA (Quantized Low-Rank Adaptation)** sprint to align model behaviors with institutional financial reporting standards, specialized syntax, and regulatory terminologies without destructive forgetting.

---

## 🛠️ Training Specifications & Hyperparameters
The model adaptation was executed using a high-efficiency **Unsloth** and **Hugging Face TRL** framework to optimize memory bounds on commercial workstation hardware.

* **Base Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
* **Fine-Tuning Method:** QLoRA (4-bit quantization via `bitsandbytes`)
* **LoRA Rank ($r$):** 16
* **LoRA Alpha ($lpha$):** 16
* **LoRA Dropout:** 0 (optimized for Unsloth stability)
* **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* **Dataset:** Curated context-response blocks containing financial balance sheets, alternative asset transaction logs, and institutional regulatory filings.
* **Optimization Pass:** AdamW 8-bit (`adamw_bnb_8bit`)
* **Learning Rate:** $2 	imes 10^{-4}$ with a linear decay schedule.

---

## 📉 Convergence Metrics
The model achieved rapid semantic convergence within a single targeted sprint, safely avoiding overfitting while forcing precise output structural syntax constraints:

| Epoch | Training Loss | Validation Loss | Perplexity | Memory Optimization |
| :--- | :--- | :--- | :--- | :--- |
| **0.5** | 1.842 | 1.621 | 5.05 | ~64% VRAM Savings |
| **1.0** | 1.215 | 1.104 | 3.01 | Via Unsloth Gradient Unrolling |
| **1.5** | 0.948 | 0.892 | 2.44 | Safe 4-bit Base Double Quant |
| **2.0** | **0.681** | **0.715** | **2.04** | Target Alignment Locked |

---

## 💾 Model Weights Tracking Disclaimer
> ⚠️ **Repository Storage and Bandwidth Constraints Notice**
> 
> Due to GitHub file-size restrictions and optimization standards for engineering workflows, the raw compiled weight serialization layers (**`.safetensors`** adapter matrix files) are **not stored directly in this Git tree**.
>
> * **Local Storage Location:** `D:dapters.zip` (Compressed snapshot containing `adapter_config.json` and `adapter_model.safetensors`).
> * **Production Note:** In a cloud deployment environment, these weights are stored securely in a private cloud registry (e.g., AWS S3, Hugging Face Private Hub, or Google Cloud Storage) and injected at runtime into the host pipeline container.

---

## 🏁 How to Review the Training Execution
1. Open the training pipeline workbook: **`financial_finetune_sprint.ipynb`**
2. The notebook contains full documented blocks detailing:
   * Dataset ingestion and tokenization padding routines.
   * Model patching via Unsloth for fast 4-bit inference/training.
   * Loss logging graphs and structural verification tests.