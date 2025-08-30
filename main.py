from fastapi import FastAPI, Query, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import google.generativeai as genai
from google import genai as direct_genai
from google.genai.types import Tool, GenerateContentConfig
from google.genai import types
import os
import pathlib
from dotenv import load_dotenv
import uvicorn
from typing import Optional
import tempfile

# 載入新的圖片路由器
from routers import image_router

# 載入環境變數
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# 檢查 API 金鑰是否設定
if not API_KEY:
    raise ValueError("GEMINI_API_KEY 環境變數未設定。請在 .env 檔案中設定此變數。")

# 配置 Gemini API
genai.configure(api_key=API_KEY)

# 初始化模型
model = genai.GenerativeModel('gemini-2.5-flash')

# 建立 FastAPI 應用
app = FastAPI(
    title="Gemini AI 服務 API",
    description="一個使用 Google Gemini 模型的多功能 API，支援 YouTube 影片摘要、文件理解、智能查詢與圖片處理。",
    version="1.1.0",
)

# 設定 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該設定為特定的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 載入圖片路由器
app.include_router(image_router.router)


# --- Pydantic 模型 ---

class VideoRequest(BaseModel):
    youtube_url: HttpUrl
    prompt: Optional[str] = "Please summarize the video. 輸出繁體中文"

class VideoSummary(BaseModel):
    summary: str

class ErrorResponse(BaseModel):
    error: str

class GroundingRequest(BaseModel):
    query: str
    use_google_search: bool = True

class GroundingResponse(BaseModel):
    summary: str

class DocumentResponse(BaseModel):
    content: str
    file_name: str

# --- 輔助函式 ---

def verify_api_key():
    """API 金鑰驗證依賴"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API 金鑰未設定")
    return True

def extract_document_content(file_path: pathlib.Path) -> Optional[str]:
    """使用 Gemini API 提取文件內容"""
    try:
        file_name = file_path.stem
        client = direct_genai.Client(api_key=API_KEY)
        
        print(f"正在上傳檔案: {file_path.name}")
        sample_file = client.files.upload(file=str(file_path))
        print(f"檔案上傳成功: {sample_file.name}")
        
        prompt = f'''
        請提取並整理這份文件的完整內容，輸出時請嚴格依照以下格式與規則：

        ## 輸出格式

        ## 文件標題
        {file_name}

        ## 文件內容
        ### 第 X 頁/章節
        - 標題：頁面/章節標題
        - 文字重點：
        - 逐條列出重點
        - 圖表/圖片說明：
        - 若有，簡要描述圖表或圖片的主要內容
        - 結論/摘要（若有）：內容

        ## 要求

        1. **逐頁提取**文件中的所有文字與重點內容。
        2. **保持原始結構**與邏輯，不刪減重要內容。
        3. **簡體中文自動轉換為繁體中文**。
        4. 使用 **Markdown 格式**輸出，確保層級分明。
        5. **保留技術術語與專有名詞**，避免誤譯或刪減。
        6. 若頁面有 **圖表或圖片**，請以文字描述其主要內容。
        7. 不需要額外的解釋或分析，只輸出整理後的內容。
        '''

        print("正在生成內容...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[sample_file, prompt]
        )
        client.files.delete(name=sample_file.name)
        print("已清理上傳的檔案")
        
        return response.text
        
    except Exception as e:
        print(f"提取文件內容時發生錯誤 ({file_path.name}): {e}")
        return None

# --- API 端點 ---

@app.get("/")
def read_root():
    """API 根路徑，返回歡迎信息"""
    return {
        "message": "歡迎使用 Gemini AI 服務 API", 
        "status": "運行中", 
        "features": [
            "YouTube 摘要", 
            "文件理解", 
            "智能查詢",
            "文字生成圖片",
            "圖片編輯"
        ]
    }

@app.get("/health")
def health_check():
    """健康檢查端點"""
    return {"status": "healthy"}

@app.post("/summarize", response_model=VideoSummary, responses={500: {"model": ErrorResponse}})
def summarize_youtube_video_post(
    request: VideoRequest,
    _: bool = Depends(verify_api_key)
):
    """
    使用 Gemini 模型摘要指定的 YouTube 影片（POST 方法）
    """
    try:
        response = model.generate_content(
            contents=[
                {"file_data": {"file_uri": str(request.youtube_url)}},
                request.prompt
            ]
        )
        return {"summary": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/grounding", response_model=GroundingResponse, responses={500: {"model": ErrorResponse}})
def grounding_query(
    request: GroundingRequest,
    _: bool = Depends(verify_api_key)
):
    """
    使用可選的 Google 搜索工具處理查詢
    """
    try:
        client = direct_genai.Client(api_key=API_KEY)
        tools = [Tool(google_search=types.GoogleSearch())] if request.use_google_search else []
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.query,
            config=GenerateContentConfig(tools=tools)
        )
        
        result_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        
        return {"summary": result_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/doc", response_model=DocumentResponse, responses={500: {"model": ErrorResponse}})
async def document_understanding(
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key)
):
    """
    理解並提取上傳文件的內容
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = pathlib.Path(tmp_file.name)
        
        try:
            extracted_content = extract_document_content(tmp_file_path)
            if extracted_content is None:
                raise HTTPException(status_code=500, detail="無法提取文件內容")
            
            return {"content": extracted_content, "file_name": file.filename}
        finally:
            if tmp_file_path.exists():
                tmp_file_path.unlink()
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 主程式執行 ---

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)