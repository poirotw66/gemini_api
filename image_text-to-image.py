from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import json
import base64
import datetime

# 載入環境變數
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# 檢查 API 金鑰是否設定
if not API_KEY:
    raise ValueError("GEMINI_API_KEY 環境變數未設定。請在 .env 檔案中設定此變數。")

client = genai.Client(api_key=API_KEY)

prompt = (
" A fresh, vibrant, and youthful girl, dressed in a Japanese-style school uniform, stands in a sun-drenched classroom. She has bright, smiling eyes and a sweet, carefree smile on her face. Her smooth, dark brown hair cascades over her shoulders, with the ends gently swaying in the breeze.",
"Her uniform is a classic white shirt paired with a dark blue blazer, and a red bow is tied at her chest. For the bottom half, she wears a plaid pleated skirt that sways just above her knees, revealing her slender, straight legs. On her feet, she wears knee-high socks and brown loafers.",
"She holds a few books in her hands and leans against the windowsill, gazing out the window. She seems to be lost in thought, perhaps waiting for the school bell to ring, ready to enjoy a wonderful afternoon with her friends.")

response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents=[prompt],
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