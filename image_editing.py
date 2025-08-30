from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import datetime
import base64
# 載入環境變數
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# 檢查 API 金鑰是否設定
if not API_KEY:
    raise ValueError("GEMINI_API_KEY 環境變數未設定。請在 .env 檔案中設定此變數。")

client = genai.Client(api_key=API_KEY)


prompt = (
"""
Using the provided image of me,西裝,油頭,背景是1940年代伯明罕街頭
"""
)
image = Image.open("image/test.jpg")

response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents=[prompt, image],
)

timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
with open(f"response_{timestamp}.txt", "w", encoding="utf-8") as f:
    f.write(str(response))

for idx, part in enumerate(response.candidates[0].content.parts):
    if part.text:
        print(part.text)
    elif part.inline_data:
        # base64 解碼
        img_data = base64.b64decode(part.inline_data.data)
        image = Image.open(BytesIO(img_data))
        filename = f"image/generated_image_{timestamp}.png"
        image.save(filename)
        print(f"✅ Image saved: {filename}")