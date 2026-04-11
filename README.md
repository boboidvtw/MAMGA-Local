# 🛸 MAMGA-Local: Multi-Graph based Agentic Memory for Local AI

**A locally-optimized implementation of the MAGMA architecture for privacy-conscious, long-term Agentic memory.**

[繁體中文版](./README-ZH.md)

[![Arxiv](https://img.shields.io/badge/Arxiv-paper-red)](https://arxiv.org/abs/2601.03236)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/boboidvtw/MAMGA-Local/actions/workflows/ci.yml/badge.svg)](https://github.com/boboidvtw/MAMGA-Local/actions/workflows/ci.yml)

## 🎯 Overview

**MAMGA-Local** is a modified distribution of the [MAGMA (MAMGA)](https://github.com/FredJiang0324/MAMGA) architecture, specifically re-engineered to work seamlessly with **Local AI** platforms like **LM Studio** and **Ollama**.

In the original research, MAGMA provides a principled multi-graph memory system for long-horizon agentic reasoning. This fork focuses on making that powerful capability accessible without relying on expensive or privacy-compromising cloud APIs.

## ✨ Key Enhancements

- **Flexible LLM Backend**: Switch between LM Studio, Ollama, and OpenAI with a single env var — no code changes needed.
- **Privacy First**: Uses local `all-MiniLM-L6-v2` for embeddings and your local hardware for reasoning. No data leaves your machine.
- **Unit Test Suite**: 74 tests covering graph DB, temporal parsing, keyword extraction, metrics, and the LLM client layer.
- **CI Pipeline**: GitHub Actions runs tests and lint on every push / pull request.
- **Bug Fixes**: Resolved Python f-string syntax errors, JSON schema compatibility issues, year extraction regex bug, and a module-level crash in `llm_judge.py` when no API key is set.
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

### 3. Configure your LLM backend

Copy `.env.example` to `.env` and edit the values:

```bash
cp .env.example .env
```

#### Option A — LM Studio (default)
```env
LLM_BACKEND=lmstudio
LLM_MODEL=local-model        # model name shown in LM Studio
# LLM_BASE_URL=http://localhost:1234/v1  # default, can omit
```
Start LM Studio, load a model, and enable the Local Server on port 1234.

#### Option B — Ollama
```env
LLM_BACKEND=ollama
LLM_MODEL=llama3
# LLM_BASE_URL=http://localhost:11434  # default, can omit
```

#### Option C — OpenAI cloud
```env
LLM_BACKEND=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

## 🚀 Quick Start

Build the memory graph and run tests using the LoCoMo dataset:
```bash
python test_fixed_memory.py --sample 0 --max-questions 5 --rebuild
```

## 🧪 Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -q
```

To skip tests that require an external LLM server:
```bash
pytest tests/ -m "not slow and not integration and not llm" -q
```

## 📂 Project Structure

```
MAMGA-Local/
├── memory/              # Core memory modules
│   ├── llm_client.py    # LLM backend abstraction (NEW)
│   ├── graph_db.py      # Four-graph structure (Temporal/Semantic/Causal/Entity)
│   ├── vector_db.py     # FAISS / NumPy vector store
│   ├── memory_builder.py
│   ├── query_engine.py
│   └── ...
├── utils/
│   └── memory_layer.py  # LLMController (env-var driven)
├── tests/               # Unit test suite (NEW)
│   ├── conftest.py
│   ├── test_graph_db.py
│   ├── test_temporal_parser.py
│   ├── test_keyword_enrichment.py
│   ├── test_evaluator.py
│   └── test_llm_client.py
├── .github/workflows/
│   └── ci.yml           # GitHub Actions CI (NEW)
├── data/                # LoCoMo dataset
├── .env.example         # Full configuration reference
└── pytest.ini
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
