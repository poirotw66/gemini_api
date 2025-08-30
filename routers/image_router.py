from fastapi import APIRouter, Query, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
from google import genai as direct_genai
from PIL import Image
from io import BytesIO
import os
import datetime
import base64

from dotenv import load_dotenv

# 載入環境變數
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# 初始化 Gemini 客戶端
# 注意：API 金鑰的檢查現在統一在 main.py 中處理
if API_KEY:
    client = direct_genai.Client(api_key=API_KEY)
else:
    client = None

# 建立路由器
router = APIRouter(
    prefix="/images",
    tags=["Image Generation & Editing"],
)

# --- Pydantic 模型 ---
class ImageGenerationRequest(BaseModel):
    prompt: str
    return_base64: bool = False

class ImageEditResponse(BaseModel):
    image_base64: Optional[str] = None
    message: str
    filename: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    image_base64: Optional[str] = None
    message: str
    filename: Optional[str] = None

# --- 輔助函式 ---

def save_and_encode_image(image_data, return_base64):
    """保存圖片並根據需要返回 base64 編碼"""
    try:
        img_data_base64 = base64.b64decode(image_data)
        image = Image.open(BytesIO(img_data_base64))
        
        # 確保 image 目錄存在
        if not os.path.exists("image"):
            os.makedirs("image")
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image/generated_image_{timestamp}.png"
        image.save(filename, "PNG")
        
        result = {
            "message": "Image generated successfully",
            "filename": filename,
            "image_base64": None
        }
        
        if return_base64:
            result["image_base64"] = image_data.decode("utf-8")
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"儲存圖片時發生錯誤: {e}")


# --- API 端點 ---

@router.post("/text-to-image", response_model=ImageGenerationResponse, summary="從文字生成圖片")
def generate_image_from_text(
    request: ImageGenerationRequest,
):
    """
    根據提供的文字提示生成一張圖片。

    - **prompt**: 用於生成圖片的詳細描述。
    - **return_base64**: 是否在回應中直接返回圖片的 base64 編碼字串。
    """
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY 未設定，無法初始化客戶端。")
    try:
        model_id = "gemini-2.5-flash-image-preview"
            
        response = client.models.generate_content(
                model=model_id,
                contents=[request.prompt],
            )    
        for idx, part in enumerate(response.candidates[0].content.parts):
            if part.text:
                print("model output = ",part.text)
            elif part.inline_data:
                result = save_and_encode_image(part.inline_data.data,request.return_base64)
                return result

        raise HTTPException(status_code=500, detail="模型未能生成圖片。")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成圖片時發生錯誤: {e}")


@router.post("/edit-image", response_model=ImageEditResponse, summary="編輯現有圖片")
async def edit_existing_image(
    prompt: str = Form(..., description="描述您想如何編輯圖片"),
    return_base64: bool = Form(False, description="是否返回 base64 編碼的圖片"),
    file: UploadFile = File(..., description="要編輯的圖片檔案"),
):
    """
    上傳一張圖片並根據文字提示進行編輯。

    - **prompt**: 描述編輯指令，例如 "幫我加上貓耳朵" 或 "背景換成海灘"。
    - **file**: 要編輯的原始圖片。
    - **return_base64**: 是否在回應中直接返回圖片的 base64 編碼字串。
    """
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY 未設定，無法初始化客戶端。")
    try:
        # 讀取上傳的圖片
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        model_id = "gemini-2.5-flash-image-preview"

        # 呼叫 Gemini API
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt, image],
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_data = part.inline_data.data
                result = save_and_encode_image(image_data, return_base64)
                result["message"] = f"圖片編輯成功。{getattr(part, 'text', '')}".strip()
                return ImageEditResponse(**result)

        raise HTTPException(status_code=500, detail="模型未能生成編輯後的圖片。")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"編輯圖片時發生錯誤: {e}")


@router.get("/download/{filename}", summary="下載生成的圖片")
def download_generated_image(filename: str):
    """
    下載先前生成或編輯的圖片。

    - **filename**: 圖片的檔案名稱。
    """
    try:
        filepath = os.path.join("image", filename)
        if os.path.exists(filepath) and filename.startswith("generated_image_"):
            return FileResponse(
                path=filepath,
                media_type="image/png",
                filename=filename
            )
        else:
            raise HTTPException(status_code=404, detail="圖片檔案不存在或路徑無效。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
