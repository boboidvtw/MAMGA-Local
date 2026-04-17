# 🛸 MAMGA-Local: Multi-Graph based Agentic Memory for Local AI

> **Memory-Augmented Multi-Agent Growth Architecture for privacy-conscious, long-term Agentic memory.**

[繁體中文版](./README-ZH.md)

### 🎯 Overview
**MAMGA-Local** is a high-performance memory architecture designed for **Local LLMs** (LM Studio, Ollama, etc.). It transforms raw unstructured data into a **Temporal Resonance Graph (TRG)**, enabling AI agents to possess persistent long-term memory and complex logical reasoning across multiple sessions.

---

### 🏗️ Architecture

1. **Recon & Research (Nomad Integration)**:
   - Acts as the "Brain" for `Project N.O.M.A.D.`.
   - Ingests real-time research reports and weaves them into the memory grid.
2. **Four-Graph Structure (TRG)**:
   - **Temporal**: Chronological event sequences.
   - **Semantic**: Concept linking via vector similarity.
   - **Causal**: Action-outcome tracking.
   - **Entity**: Subject mapping across sessions.
3. **Query Engine**:
   - Optimized for **Multi-hop Reasoning** and temporal logic queries.

---

### 📥 Data Sources
- **Nomad Reports**: Automated ingestion via the Nomad-Navigator agent.
- **Structured Data**: Supports LoCoMo JSON and local Markdown archives.
- **Memory Layers**: Uses `faiss-cpu` for vector search and `networkx` for graph reasoning.

---

### ⚙️ Installation

1. **Requirements**: Python 3.9+, numpy, faiss-cpu, torch.
2. **Setup**:
   ```bash
   git clone https://github.com/boboidvtw/MAMGA-Local.git
   cd MAMGA-Local
   pip install -r requirements.txt
   ```
3. **Configuration**: Copy `.env.example` to `.env` and set `LLM_BACKEND=lmstudio`.

---

### 🚀 Usage

1. **Build Memory**:
   ```bash
   python3 main.py build --input path/to/data.json
   ```
2. **Query Memory**:
   ```bash
   python3 main.py query --question "What are the core findings of the recent research?"
   ```
3. **Nomad Integration**:
   This module is automatically triggered by Nomad's navigation scripts.

---

### 📜 Citation & Credits
Localized implementation of the MAGMA architecture.
```bibtex
@article{jiang2026magma,
  title={MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents},
  author={Jiang, Dongming and Li, Yi and Li, Guanpeng and Li, Bingzhe},
  year={2026}
}
```

### 📄 License
MIT License.
