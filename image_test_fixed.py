from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# 檢查 API 金鑰是否設定
if not API_KEY:
    raise ValueError("GEMINI_API_KEY 環境變數未設定。請在 .env 檔案中設定此變數。")

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

# 正確的 Client 初始化方式
client = genai.Client(api_key=API_KEY)

prompt = (
    "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme"
)

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[prompt],
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print("生成的文字描述:")
            print(part.text)
        elif part.inline_data is not None:
            print(f"檢測到圖片數據，MIME 類型: {part.inline_data.mime_type}")
            print(f"數據類型: {type(part.inline_data.data)}")
            
            # 處理圖片數據
            if isinstance(part.inline_data.data, bytes):
                image_data = part.inline_data.data
            else:
                print(f"未預期的數據類型: {type(part.inline_data.data)}")
                continue
                
            try:
                image = Image.open(BytesIO(image_data))
                print(f"圖片格式: {image.format}, 尺寸: {image.size}")
                image.save("generated_image.png")
                print("圖片已保存為 generated_image.png")
            except Exception as img_error:
                print(f"處理圖片時發生錯誤: {img_error}")
                
except Exception as e:
    print(f"生成圖片時發生錯誤: {e}")
