from fastapi import FastAPI, Query, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import google.generativeai as genai
from google import genai as direct_genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from google.genai import types
import os
import pathlib
from dotenv import load_dotenv
import uvicorn
from typing import Optional, List, Dict
import tempfile
from PIL import Image
from io import BytesIO
import base64

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
    description="一個使用 Google Gemini 模型的多功能 API，支援 YouTube 影片摘要、文件理解和智能查詢",
    version="1.0.0",
)

# 設定 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該設定為特定的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定義請求模型
class VideoRequest(BaseModel):
    youtube_url: HttpUrl
    prompt: Optional[str] = "Please summarize the video. 輸出繁體中文"

# 定義回應模型
class VideoSummary(BaseModel):
    summary: str

class ErrorResponse(BaseModel):
    error: str

# 定義 Grounding 請求和回應模型
class GroundingRequest(BaseModel):
    query: str
    use_url_context: bool = True
    use_google_search: bool = True

class GroundingResponse(BaseModel):
    summary: str

# 定義文件理解請求和回應模型
class DocumentRequest(BaseModel):
    file_name: Optional[str] = None

class DocumentResponse(BaseModel):
    content: str
    file_name: str

# 定義圖片生成請求和回應模型
class ImageGenerationRequest(BaseModel):
    prompt: str
    return_base64: bool = False  # 是否返回 base64 編碼的圖片

class ImageGenerationResponse(BaseModel):
    image_base64: Optional[str] = None
    message: str
    filename: Optional[str] = None

# 檢查 API 金鑰函數
def verify_api_key():
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API 金鑰未設定")
    return True

def extract_ppt_content(file_path: pathlib.Path) -> Optional[str]:
    """使用 Gemini API 提取 PPT 內容"""
    try:
        file_name = file_path.stem
        client = direct_genai.Client(api_key=API_KEY)
        model_id = "gemini-2.5-flash"
        
        # 上傳檔案到 Gemini
        print(f"正在上傳檔案: {file_path.name}")
        sample_file = client.files.upload(file=str(file_path))
        print(f"檔案上傳成功: {sample_file.name}")
        
        # 提取內容的 prompt
        prompt = f"""
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
        """

        # 呼叫 Gemini API
        print("正在生成內容...")
        # 呼叫 Gemini API
        response = client.models.generate_content(
            model=model_id,
            contents=[sample_file, prompt]
        )
        # 清理上傳的檔案
        client.files.delete(name=sample_file.name)
        print("檔案已清理")
        
        return response.text
        
    except Exception as e:
        print(f"提取文件內容時發生錯誤 ({file_path.name}): {e}")
        return None

def generate_image(prompt: str, return_base64: bool = False) -> dict:
    """使用 Gemini API 生成圖片"""
    try:
        client = direct_genai.Client(api_key=API_KEY)
        model_id = "gemini-2.5-flash-image-preview"
        
        print(f"正在生成圖片，提示: {prompt}")
        
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt],
        )
        
        # 處理回應
        generated_text = ""
        image_data = None
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                generated_text += part.text
                print(f"生成的文字描述: {part.text}")
            elif part.inline_data is not None:
                print(f"檢測到圖片數據，MIME 類型: {part.inline_data.mime_type}")
                print(f"數據類型: {type(part.inline_data.data)}")
                
                # 處理不同類型的數據
                if isinstance(part.inline_data.data, bytes):
                    image_data = part.inline_data.data
                elif isinstance(part.inline_data.data, str):
                    # 如果是 base64 字符串，先解碼
                    try:
                        image_data = base64.b64decode(part.inline_data.data)
                    except Exception as decode_error:
                        print(f"無法解碼 base64 數據: {decode_error}")
                        continue
                else:
                    print(f"未知的數據格式: {type(part.inline_data.data)}")
                    continue
                
                print("圖片數據處理成功")
        
        if image_data:
            try:
                # 驗證圖片數據
                image = Image.open(BytesIO(image_data))
                print(f"圖片格式: {image.format}, 尺寸: {image.size}")
                
                # 將圖片保存到臨時檔案
                import time
                filename = f"generated_image_{int(time.time())}.png"
                image.save(filename, "PNG")
                print(f"圖片已保存為: {filename}")
                
                result = {
                    "message": f"圖片生成成功。{generated_text}",
                    "filename": filename
                }
                
                if return_base64:
                    # 如果需要返回 base64 編碼
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    result["image_base64"] = image_base64
                
                return result
                
            except Exception as img_error:
                print(f"處理圖片時發生錯誤: {img_error}")
                return {
                    "message": f"圖片數據處理失敗: {str(img_error)}。文字回應: {generated_text}",
                    "filename": None
                }
        else:
            return {
                "message": f"未能生成圖片，但有文字回應: {generated_text}",
                "filename": None
            }
        
    except Exception as e:
        print(f"生成圖片時發生錯誤: {e}")
        return {
            "message": f"生成圖片時發生錯誤: {str(e)}",
            "filename": None
        }

@app.get("/")
def read_root():
    """API 根路徑，返回歡迎信息"""
    return {"message": "歡迎使用 Gemini AI 服務 API", "status": "運行中", "features": ["YouTube摘要", "文件理解", "智能查詢", "圖片生成"]}

@app.get("/health")
def health_check():
    """健康檢查端點"""
    return {"status": "healthy"}

@app.get("/summarize", response_model=VideoSummary, responses={500: {"model": ErrorResponse}})
def summarize_youtube_video(
    url: str = Query(..., description="YouTube 視頻連結"),
    _: bool = Depends(verify_api_key)
):
    """
    使用 Gemini 模型摘要指定的 YouTube 影片（繁體中文）
    
    - **url**: YouTube 影片網址
    
    回傳:
    - **summary**: 影片的中文摘要
    """
    try:
        response = model.generate_content(
            contents=[
                {"file_data": {"file_uri": url}},
                "Please summarize the video. 輸出繁體中文"
            ]
        )
        return {"summary": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize", response_model=VideoSummary, responses={500: {"model": ErrorResponse}})
def summarize_youtube_video_post(
    request: VideoRequest,
    _: bool = Depends(verify_api_key)
):
    """
    使用 Gemini 模型摘要指定的 YouTube 影片（POST 方法）
    
    - **request**: 包含 YouTube 網址和可選提示的請求
    
    回傳:
    - **summary**: 影片的中文摘要
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
    使用 Gemini 模型處理查詢，可選使用 URL 上下文和 Google 搜索工具進行資訊檢索
    
    - **request**: 包含查詢和工具使用選項的請求
    
    回傳:
    - **response**: Gemini 的回應
    - **url_context_metadata**: URL 上下文元數據（如果使用了 URL 上下文工具）
    """
    try:
        # 使用直接客戶端方式
        client = direct_genai.Client(api_key=API_KEY)
        model_id = "gemini-2.5-flash"
        
        tools = []
        # if request.use_url_context:
        #     tools.append(Tool(url_context=types.UrlContext))
        if request.use_google_search:
            tools.append(Tool(google_search=types.GoogleSearch))
        
        response = client.models.generate_content(
            model=model_id,
            contents=request.query,
            config=GenerateContentConfig(
                tools=tools,
                response_modalities=["TEXT"],
            )
        )
        
        # 處理回應
        result_text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text'):
                result_text += part.text
        
        return {
            "summary": result_text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image_generation", response_model=ImageGenerationResponse, responses={500: {"model": ErrorResponse}})
def generate_image_get(
    prompt: str = Query(..., description="圖片生成提示"),
    return_base64: bool = Query(False, description="是否返回 base64 編碼的圖片"),
    _: bool = Depends(verify_api_key)
):
    """
    使用 Gemini 模型生成圖片（GET 方法）
    
    - **prompt**: 圖片生成的提示詞
    - **return_base64**: 是否返回 base64 編碼的圖片資料
    
    回傳:
    - **message**: 生成結果訊息
    - **filename**: 生成的圖片檔案名稱（如果成功）
    - **image_base64**: base64 編碼的圖片資料（如果 return_base64=true）
    """
    try:
        result = generate_image(prompt, return_base64)
        return ImageGenerationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image_generation", response_model=ImageGenerationResponse, responses={500: {"model": ErrorResponse}})
def generate_image_post(
    request: ImageGenerationRequest,
    _: bool = Depends(verify_api_key)
):
    """
    使用 Gemini 模型生成圖片（POST 方法）
    
    - **request**: 包含圖片生成提示和選項的請求
    
    回傳:
    - **message**: 生成結果訊息
    - **filename**: 生成的圖片檔案名稱（如果成功）
    - **image_base64**: base64 編碼的圖片資料（如果 return_base64=true）
    """
    try:
        result = generate_image(request.prompt, request.return_base64)
        return ImageGenerationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_image/{filename}")
def download_generated_image(filename: str):
    """
    下載生成的圖片檔案
    
    - **filename**: 圖片檔案名稱
    """
    try:
        if os.path.exists(filename) and filename.startswith("generated_image_"):
            return FileResponse(
                path=filename,
                media_type="image/png",
                filename=filename
            )
        else:
            raise HTTPException(status_code=404, detail="圖片檔案不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/doc", response_model=DocumentResponse, responses={500: {"model": ErrorResponse}})
async def document_understanding(
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key)
):
    """
    使用 Gemini 模型理解並提取文件內容（支援 PPT、PDF 等格式）
    
    - **file**: 上傳的文件（PPT、PDF 等）
    
    回傳:
    - **content**: 提取並整理後的文件內容
    - **file_name**: 檔案名稱
    """
    try:
        # 創建臨時檔案
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = pathlib.Path(tmp_file.name)
        
        try:
            # 提取內容
            extracted_content = extract_ppt_content(tmp_file_path)
            
            if extracted_content is None:
                raise HTTPException(status_code=500, detail="無法提取文件內容")
            
            return {
                "content": extracted_content,
                "file_name": file.filename
            }
        finally:
            # 清理臨時檔案
            if tmp_file_path.exists():
                tmp_file_path.unlink()
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 啟動應用
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
