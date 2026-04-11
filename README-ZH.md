# 🛸 MAMGA-Local: 多圖譜智能代理本地記憶架構

**一個針對本地 AI 環境優化、基於學術研究成果的 Agent 長期記憶系統。**

[![Arxiv](https://img.shields.io/badge/Arxiv-paper-red)](https://arxiv.org/abs/2601.03236)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/boboidvtw/MAMGA-Local/actions/workflows/ci.yml/badge.svg)](https://github.com/boboidvtw/MAMGA-Local/actions/workflows/ci.yml)

## 🎯 專案目標

MAMGA-Local 是對原始 [MAGMA (MAMGA)](https://github.com/FredJiang0324/MAMGA) 行為架構的改進版本，旨在讓開發者與研究者能在**完全隱私、無雲端依賴**的情況下，為其 AI Agents 建立強大的長短期記憶能力。

本專案將複雜的「多圖譜」記憶理論與 **LM Studio / Ollama** 等本地模型平台深度整合，實現了「去雲端化」的高性能記憶檢索。

## ✨ 核心特性

- **彈性 LLM 後端**：只需修改一個 env var，即可在 LM Studio、Ollama 和 OpenAI 之間切換，無需修改任何程式碼。
- **100% 隱私與本地化**：預設使用本機 `all-MiniLM-L6-v2` Embedding，所有資料均不離開本機。
- **完整單元測試**：74 個測試覆蓋圖資料庫、時間解析、關鍵字萃取、評估指標與 LLM client 層。
- **CI 自動化流程**：每次 push / PR 自動執行測試與 lint（GitHub Actions）。
- **多圖譜推理 (Multi-Graph Reasoning)**：自動構建並維護四種正交關係圖譜：
  - **語意 (Semantic)**：內容相關連。
  - **時間 (Temporal)**：事件發生順序。
  - **因果 (Causal)**：行為與結果的關聯。
  - **實體 (Entity)**：跨對話的物件追蹤。
- **Bug 修復**：修正 f-string 編碼錯誤、年份 regex 擷取 bug、`llm_judge.py` import 時崩潰等問題。

## 🛠️ 安裝與設置

### 1. 克隆倉庫
```bash
git clone https://github.com/boboidvtw/MAMGA-Local.git
cd MAMGA-Local
```

### 2. 環境配置 (Python 3.9+)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置 LLM 後端

將 `.env.example` 複製為 `.env` 並填入設定：

```bash
cp .env.example .env
```

#### 選項 A — LM Studio（預設）
```env
LLM_BACKEND=lmstudio
LLM_MODEL=local-model        # LM Studio 中顯示的模型名稱
# LLM_BASE_URL=http://localhost:1234/v1  # 預設值，可省略
```
啟動 LM Studio，載入模型，並在 Local Server 頁面開啟 port 1234。

#### 選項 B — Ollama
```env
LLM_BACKEND=ollama
LLM_MODEL=llama3
# LLM_BASE_URL=http://localhost:11434  # 預設值，可省略
```

#### 選項 C — OpenAI 雲端
```env
LLM_BACKEND=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

## 🚀 快速啟動

使用 LoCoMo 數據集進行記憶構建與問答測試：
```bash
python test_fixed_memory.py --sample 0 --max-questions 5 --rebuild
```

## 🧪 執行測試

```bash
pip install pytest pytest-cov
pytest tests/ -q
```

跳過需要外部 LLM 服務的測試：
```bash
pytest tests/ -m "not slow and not integration and not llm" -q
```

## 📂 專案結構

```
MAMGA-Local/
├── memory/              # 核心記憶模組
│   ├── llm_client.py    # LLM 後端抽象層（新增）
│   ├── graph_db.py      # 四圖結構（時間/語意/因果/實體）
│   ├── vector_db.py     # FAISS / NumPy 向量庫
│   ├── memory_builder.py
│   ├── query_engine.py
│   └── ...
├── utils/
│   └── memory_layer.py  # LLMController（已改為 env var 驅動）
├── tests/               # 單元測試套件（新增）
│   ├── conftest.py
│   ├── test_graph_db.py
│   ├── test_temporal_parser.py
│   ├── test_keyword_enrichment.py
│   ├── test_evaluator.py
│   └── test_llm_client.py
├── .github/workflows/
│   └── ci.yml           # GitHub Actions CI（新增）
├── data/                # LoCoMo 數據集
├── .env.example         # 完整配置說明
└── pytest.ini
```

## 📣 致謝與引用

原始理論架構參考自論文：
[MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](https://arxiv.org/abs/2601.03236)

```bibtex
@article{jiang2026magma,
  title={MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents},
  author={Jiang, Dongming and Li, Yi and Li, Guanpeng and Li, Bingzhe},
  journal={arXiv preprint arXiv:2601.03236},
  year={2026}
}
```

## 📄 授權協議

本專案採用 [MIT License](LICENSE)。
