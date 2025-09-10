import fitz
import pytesseract
import cv2
import numpy as np
import io
import re
from PIL import Image
from pathlib import Path



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"





def ocr_image(img: np.ndarray, isExist: bool = True) -> str:
    """이미지에서 OCR 수행, 숫자/한글 위주로"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if isExist:
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        # Tesseract로 숫자 + 한글 추출
        text = pytesseract.image_to_string(
            thresh,
            lang='kor'
        )
    else:
        gray = cv2.medianBlur(gray, 3)
        thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # Tesseract로 숫자 + 한글 추출
        text = pytesseract.image_to_string(
            thresh,
            lang='kor'
        )
    

    return text

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""

    for page in doc:
        text = page.get_text().strip()
        pix = page.get_pixmap(dpi=300)  # PDF → 이미지
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n > 3:  # alpha 채널 제거
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        ocr_txt  = ocr_image(img)
        if not ocr_txt :
            ocr_txt  = ocr_image(img, False)

        combined = (text + "\n" + ocr_txt).strip()
        full_text += combined + "\n"

    return full_text



def extract_text_from_img(file_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(file_bytes))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    full_text = ocr_image(img)
    if not full_text:
        full_text = ocr_image(img, False)

    return full_text


def extract_business_info(text: str) -> dict:
    business_number = re.findall(r'\d{3}-\d{2}-\d{5}', text)
    # 간단한 예시: "상호", "대표" 키워드 뒤 텍스트 추출
    name_match = re.search(r'(?:상\s*호|법인명|단체명)\s*[:：~]?\s*(.+)', text)
    ceo_match = re.search(r'(?:대표자|성\s*명)\s*[:：~]?\s*(.+)', text)

    return {
        "사업자번호": business_number[0] if business_number else None,
    }

folder_path = r"D:\LearningPython\OCR\test_pdfs"  # 처리할 PDF/이미지 폴더
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

