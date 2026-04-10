# 🛸 MAMGA-Local: 多圖譜智能代理本地記憶架構

**一個針對本地 AI 環境優化、基於學術研究成果的 Agent 長期記憶系統。**

[![Arxiv](https://img.shields.io/badge/Arxiv-paper-red)](https://arxiv.org/abs/2601.03236)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 專案目標

MAMGA-Local 是對原始 [MAGMA (MAMGA)](https://github.com/FredJiang0324/MAMGA) 行為架構的改進版本，旨在讓開發者與研究者能在**完全隱私、無雲端依賴**的情況下，為其 AI Agents 建立強大的長短期記憶能力。

本專案將複雜的「多圖譜」記憶理論與 **LM Studio / Ollama** 等本地模型平台深度整合，實現了「去雲端化」的高性能記憶檢索。

## ✨ 核心特性

- **100% 隱私與本地化**：預設支援 `http://localhost:1234/v1` (LM Studio)，支援 0 元成本運行。
- **多圖譜推理 (Multi-Graph Reasoning)**：自動構建並維護四種正交關係圖譜：
  - **語意 (Semantic)**：內容相關連。
  - **時間 (Temporal)**：事件發生順序。
  - **因果 (Causal)**：行為與結果的關聯。
  - **實體 (Entity)**：跨對話的物件追蹤。
- **混合檢索引擎**：整合了 FAISS/ChromaDB 向量檢索與 NetworkX 圖論遍歷。
- **中文支援優化**：修復了原始脚本在中文環境下可能遇到的 f-string 編碼與格式錯誤。

## 🛠️ 安裝與設置

### 1. 克隆倉庫
```bash
git clone https://github.com/boboidvtw/MAMGA-Local.git
cd MAMGA-Local
```

### 2. 環境配置 (Python 3.9+)
```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
```

### 3. 配置本地模型 (以 LM Studio 為例)
1. 啟動 **LM Studio** 並載入一個模型（如 Llama 3 或 Gemma 3）。
2. 在 **Local Server** 頁面啟動伺服器（預設為 `localhost:1234`）。
3. 建立 `.env` 檔案：
```bash
OPENAI_API_KEY=lm-studio
OPENAI_BASE_URL=http://localhost:1234/v1
MODEL_NAME=gpt-4  # 雖然這裡是寫 gpt-4，但會連向你的本地模型
```

## 🚀 快速啟動

使用 LoCoMo 數據集進行記憶構建與問答測試：
```bash
# 執行測試（將自動調用本地模型進行記憶蒸餾）
python test_fixed_memory.py --sample 0 --max-questions 5 --rebuild
```

## 📂 文件結構

- `memory/`: 核心記憶模組（圖譜構建、向量庫）。
- `utils/memory_layer.py`: LLM 控制器（已針對本地 API 優化）。
- `data/`: 包含預載的 LoCoMo 數據集。
- `test_fixed_memory.py`: 主要的效能評測與演示腳本。

## 📣 致謝與引用

原始理論架構參考自論文：
[MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](https://arxiv.org/abs/2601.03236)

## 📄 授權協議

本專案採用 [MIT License](LICENSE)。
