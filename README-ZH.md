# 🧠 MAMGA-Local: 本地 AI 長期記憶增強架構

> **Memory-Augmented Multi-Agent Growth Architecture for Local AI**

這是一個專為**本地/離線 AI** 打造的長期記憶系統。它將對話、研究報告與文獻轉化為一個具備邏輯聯結的「四維記憶圖譜 (TRG)」，使 AI 能夠具備跨越時間與文檔的推理能力。

---

### 🎯 專案目標 (Project Goals)
- **消除 AI 的遺忘**: 為本地 LLM 提供持久化的知識儲存，解決上下文視窗限制。
- **邏輯聯結**: 不只是向量檢索 (RAG)，而是建立事件之間的時序、語義、因果與實體關聯。
- **隱私優先**: 所有的記憶處理與圖譜構建均在本地執行，數據不外流。

---

### 🏗️ 專案架構 (Architecture)

1. **記憶建構器 (MemoryBuilder)**:
   - 提取對話中的關鍵事實、實體與關係。
   - 基於 **Temporal Resonance Graph (TRG)** 建立四種維度的鏈結。
2. **查詢引擎 (QueryEngine)**:
   - 支援「多跳推理 (Multi-hop Reasoning)」，能回答像「A 事件發生後，對 B 產生了什麼後續影響？」這類複雜問題。
3. **專屬組件**:
   - `TRG Memory`: 核心圖形資料庫結構。
   - `LLMController`: 自動偵測與調度本地模型（LM Studio, Ollama, vLLM 等）。
4. **Nomad 橋接器**:
   - 專為 `Project N.O.M.A.D.` 設計的數據接口，支持自動編織 Markdown 報告進記憶網格。

---

### 📥 資料來源與引用 (Data Sources)
- **Nomad 研究報告**: 接收來自 `Nomad-Navigator` 的實時研究內容。
- **LoCoMo 數據集**: 支持標準的長時序對話基準測試。
- **本地文獻**: 可解析 Markdown、JSON 或 Text 檔案進行知識建構。

---

### ⚙️ 安裝說明 (Installation)

1. **環境需求**: Python 3.9+ (推薦 3.12+), FAISS-cpu 1.7.4+。
2. **安裝步驟**:
   ```bash
   git clone https://github.com/boboidvtw/MAMGA-Local.git
   cd MAMGA-Local
   pip install -r requirements.txt
   ```
3. **環境配置**: 複製 `.env.example` 到 `.env` 並設定你的 `LLM_BACKEND`。

---

### 🚀 使用說明 (Usage)

1. **建立記憶圖譜**:
   ```bash
   # 指向你的報告目錄或數據集進行編織
   python3 main.py build --input data/locomo10.json
   ```
2. **執行智慧查詢**:
   ```bash
   # 進行邏輯推理查詢
   python3 main.py query --question "描述專案演進的三大關鍵點"
   ```
3. **Nomad 整合測試**:
   直接執行 Nomad 的 `browser_navigator.py`，記憶會自動同步至此處。

---

### 📊 學術引用 (Citation)
本專案為 MAGMA 架構的本地化實作。
```bibtex
@article{jiang2026magma,
  title={MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents},
  year={2026}
}
```

### 🛠️ 維護者
*   **開發者**：[boboidvtw](https://github.com/boboidvtw)
*   **授權**：MIT License
