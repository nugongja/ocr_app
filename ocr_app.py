import easyocr
import fitz  
import cv2
import numpy as np
import io
from PIL import Image
from pathlib import Path
import re

reader = easyocr.Reader(['ko'])


# --------------------------
# OCR 수행 함수
# --------------------------
def ocr_image(img: np.ndarray) -> str:
    """이미지에서 OCR 수행 후 텍스트 합치기"""
    results = reader.readtext(img)
    text = " ".join([res[1] for res in results])
    # 숫자 사이 공백 제거 (667 - 82 - 00245 → 667-82-00245)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    return text

# --------------------------
# PDF에서 텍스트 추출
# --------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""

    for page in doc:
        # PDF → 이미지 변환 (dpi 300으로 품질 확보)
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n > 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR, cv2.COLOR_RGBA2BGR)
        
        text = ocr_image(img)
        full_text += text + "\n"

    return full_text

# --------------------------
# 이미지에서 텍스트 추출
# --------------------------
def extract_text_from_img(file_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(file_bytes))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    text = ocr_image(img)
    return text

# --------------------------
# 사업자번호 추출
# --------------------------
def extract_business_info(text: str) -> dict:
    pattern = r'(\d{3})\s*-\s*(\d{2})\s*-\s*(\d{5})'
    match = re.search(pattern, text)
    business_number = "-".join(match.groups()) if match else None
    return {"사업자번호": business_number}


# --------------------------
# 폴더 내 PDF/이미지 처리
# --------------------------
folder_path = r"D:\LearningPython\OCR\test_pdfs"
results = []

for file_path in Path(folder_path).glob("*.*"):
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    if file_path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(file_bytes)
    else:
        text = extract_text_from_img(file_bytes)

    info = extract_business_info(text)
    info["filename"] = file_path.name
    results.append(info)

# 결과 출력
for r in results:
    print(r)
