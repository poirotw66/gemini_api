# YouTube Gemini 摘要 API

一個使用 Google Gemini 模型來摘要 YouTube 影片的 FastAPI 應用。

## 功能

- 提供 GET 和 POST API 端點來摘要 YouTube 影片
- 使用 Gemini 2.5 Flash 模型進行影片分析
- 以繁體中文輸出摘要結果
- 支援 CORS
- 提供健康檢查端點

## 安裝

1. 克隆此倉庫：

```bash
git clone <repository-url>
cd fast_api
```

2. 安裝依賴：

```bash
pip install -r requirements.txt
```

3. 設定環境變數：

複製 example.env 到 .env 並填入您的 Gemini API 金鑰：

```bash
cp example.env .env
```

然後編輯 .env 文件，填入您的 Gemini API 金鑰。

## 使用方法

啟動伺服器：

```bash
python main.py
```

或者使用 uvicorn：

```bash
uvicorn main:app --reload
```

服務將在 http://localhost:8000 上啟動。

## API 端點

### GET /summarize

參數：
- `url`: YouTube 影片網址

範例：
```
GET /summarize?url=https://www.youtube.com/watch?v=S7ARexSGEGo
```

### POST /summarize

請求體：
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=S7ARexSGEGo",
  "prompt": "Please summarize the video. 輸出繁體中文"
}
```

### GET /health

用於健康檢查的端點。

### POST /doc
curl -X POST "http://localhost:8001/doc" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test.pdf"

## API 文檔

訪問 http://localhost:8000/docs 可獲取完整的 Swagger UI 文檔。

## 注意事項

請確保您已經獲取了有效的 Gemini API 金鑰，並妥善保管。不要將包含 API 金鑰的 .env 檔案提交到版本控制系統。
