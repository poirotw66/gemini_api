# Gemini AI 多功能服務 API

這是一個使用 Google Gemini 模型打造的多功能後端服務，基於 FastAPI 框架開發。它不僅提供強大的 AI 功能，還具備高擴展性和模組化的架構。

## ✨ 主要功能

- **📝 YouTube 影片摘要**: 輸入 YouTube 連結，快速生成影片的繁體中文摘要。
- **📄 文件理解**: 上傳文件（如 PDF、PPT），API 會提取並整理其核心內容。
- **🌐 智能問答 (Grounding)**: 結合 Google 搜索，提供更具事實基礎的問答體驗。
- **🎨 文字生成圖片**: 根據文字描述創造出獨特的圖片。
- **🖼️ 圖片編輯**: 上傳一張圖片，並透過文字指令對其進行修改。

## 🚀 快速啟動

1.  **克隆專案**:
    ```bash
    git clone <repository-url>
    cd gemini_api
    ```

2.  **安裝依賴**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **設定環境變數**:
    在專案根目錄下建立一個 `.env` 檔案，並填入您的 Gemini API 金鑰：
    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

4.  **啟動服務**:
    ```bash
    python main.py
    ```
    服務將在 `http://localhost:8001` 上運行。

## 📚 API 端點說明

您可以訪問 `http://localhost:8001/docs` 來查看完整的 Swagger UI 互動式 API 文件。

---

### 影片與文件處理

-   **`POST /summarize`**: 摘要指定的 YouTube 影片。
    -   **Request Body**: `{"youtube_url": "...", "prompt": "..."}`
-   **`POST /doc`**: 上傳並分析文件內容。
    -   **Request Body**: `multipart/form-data`，包含一個 `file` 欄位。

### 智能查詢

-   **`POST /grounding`**: 根據提供的查詢進行 Google 搜索並生成回覆。
    -   **Request Body**: `{"query": "...", "use_google_search": true}`

### 圖片處理 (前綴: `/images`)

-   **`POST /images/text-to-image`**: 根據文字提示生成圖片。
    -   **Request Body**: `{"prompt": "A cat wearing a hat", "return_base64": false}`
-   **`POST /images/edit-image`**: 上傳圖片並根據文字指令進行修改。
    -   **Request Body**: `multipart/form-data`，包含 `prompt` (文字指令) 和 `file` (圖片檔案) 兩個欄位。
-   **`GET /images/download/{filename}`**: 下載先前生成或編輯過的圖片。

### 系統

-   **`GET /`**: API 根目錄，顯示歡迎訊息和功能列表。
-   **`GET /health`**: 健康檢查端點，確認服務是否正常運行。

## ⚠️ 注意事項

-   請確保您的 `GEMINI_API_KEY` 是有效且保密的。
-   `.gitignore` 檔案已設定忽略 `.env` 檔案，請勿將其提交到版本控制系統中。