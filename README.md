# 🛸 MAMGA-Local: Multi-Graph based Agentic Memory for Local AI

**A locally-optimized implementation of the MAGMA architecture for privacy-conscious, long-term Agentic memory.**

[繁體中文版](./README-ZH.md)

[![Arxiv](https://img.shields.io/badge/Arxiv-paper-red)](https://arxiv.org/abs/2601.03236)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

**MAMGA-Local** is a modified distribution of the [MAGMA (MAMGA)](https://github.com/FredJiang0324/MAMGA) architecture, specifically re-engineered to work seamlessly with **Local AI** platforms like **LM Studio** and **Ollama**.

In the original research, MAGMA provides a principled multi-graph memory system for long-horizon agentic reasoning. This fork focuses on making that powerful capability accessible without relying on expensive or privacy-compromising cloud APIs.

## ✨ Key Enhancements

- **Local LLM Integration**: Pre-configured to work with `localhost:1234/v1` (LM Studio).
- **Privacy First**: Uses local `all-MiniLM-L6-v2` for embeddings and your local hardware for reasoning. No data leaves your machine.
- **Bug Fixes**: Resolved Python f-string syntax errors and JSON schema compatibility issues commonly found when using local inference servers.
- **Bilingual Documentation**: Support for both English and Traditional Chinese.

## 🛠️ Installation

### 1. Clone & Setup
```bash
git clone https://github.com/boboidvtw/MAMGA-Local.git
cd MAMGA-Local
```

### 2. Environment (Python 3.9+)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Local Model Setup
1. Open **LM Studio** and load your preferred model (e.g., Llama 3, Gemma 3).
2. Start the **Local Server** (typically on port 1234).
3. Create your `.env`:
```bash
OPENAI_API_KEY=lm-studio
OPENAI_BASE_URL=http://localhost:1234/v1
MODEL_NAME=gpt-4  # Requests are redirected to your local model
```

## 🚀 Quick Start

Build the memory graph and run tests using the LoCoMo dataset:
```bash
python test_fixed_memory.py --sample 0 --max-questions 5 --rebuild
```

## 📖 Architecture & Theory

This system maintains four orthogonal graph relationships to organize conversation memory:
- **Temporal**: Links events in chronological order.
- **Semantic**: Connects related concepts via vector similarity.
- **Causal**: Tracks actions and their outcomes.
- **Entity**: Maps recurring subjects across different sessions.

## 📣 Citation & Credits

This project is a localized implementation of the MAGMA architecture. Please cite the original paper if you use this in your research:

```bibtex
@article{jiang2026magma,
  title={MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents},
  author={Jiang, Dongming and Li, Yi and Li, Guanpeng and Li, Bingzhe},
  journal={arXiv preprint arXiv:2601.03236},
  year={2026}
}
```

## 📄 License

MIT License.
