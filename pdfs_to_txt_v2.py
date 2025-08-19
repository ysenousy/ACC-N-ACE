import os
import fitz  
import pdfplumber
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract
import re
import cv2

# Set your folders
pdf_folder = "PDFs"
output_folder = "extracted_papers"
images_folder = os.path.join(output_folder, "images")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)

# Function: Extract text using PDFMiner
def extract_text_pdfminer(pdf_path):
    try:
        return extract_text(pdf_path)
    except Exception:
        return ""

# Function: Extract text using OCR on scanned PDF pages
def extract_text_ocr(pdf_path):
    try:
        pages = convert_from_path(pdf_path, dpi=300)
        return '\n'.join(pytesseract.image_to_string(page, lang='eng') for page in pages)
    except:
        return ""

# Function: Extract tables using pdfplumber
def extract_tables(pdf_path):
    tables_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        tables_text += "\t".join(str(cell) if cell else "" for cell in row) + "\n"
                    tables_text += "\n"
    except Exception:
        tables_text += "[Error extracting tables]\n"
    return tables_text

# Function: Extract images using PyMuPDF
def extract_images(pdf_path, pdf_name):
    images_info = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"{os.path.splitext(pdf_name)[0]}_page{i+1}_img{img_index+1}.{image_ext}"
                image_path = os.path.join(images_folder, image_filename)

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                images_info.append(image_filename)
    except Exception:
        images_info.append("[Error extracting images]")
    return images_info

# Function: Extract text from the saved images using OCR
def extract_text_from_images(image_filenames):
    ocr_text = ""
    for image_file in image_filenames:
        image_path = os.path.join(images_folder, image_file)
        try:
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            # Preprocess: grayscale + threshold
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # OCR
            text = pytesseract.image_to_string(thresh, lang='eng', config='--oem 3 --psm 6')
            
            # Clean text
            cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,%-]', '', text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            if len(cleaned_text) > 10:  # Meaningful text
                ocr_text += f"\n--- Text from {image_file} ---\n{cleaned_text}\n"
            else:
                # Delete empty images
                os.remove(image_path)
                print(f"Deleted {image_file} (no text found)")
                
        except Exception as e:
            print(f"[Error processing {image_file}]: {e}")
    return ocr_text

def clean_text(text):
    # Remove multiple spaces, line breaks, tabs
    text = re.sub(r'\s+', ' ', text)
    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode()
    # Optionally, lowercasing
    text = text.lower()
    return text.strip()

# Main loop to process all PDFs
for filename in os.listdir(pdf_folder):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_folder, filename)
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(output_folder, txt_filename)

    print(f"[*] Processing {filename}...")

    # Extract main content
    text = extract_text_pdfminer(pdf_path)
    if not text.strip():
        print("   [!] PDFMiner failed → using OCR")
        text = extract_text_ocr(pdf_path)
    text = clean_text(text)

    tables = extract_tables(pdf_path)
    tables = clean_text(tables)

    images = extract_images(pdf_path, filename)

    image_ocr_text = extract_text_from_images(images)
    image_ocr_text = clean_text(image_ocr_text)

    # Write everything to .txt file
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("==== TEXT ====\n")
        f.write(text + "\n\n")

        f.write("==== TABLES ====\n")
        f.write(tables + "\n")

        f.write("==== IMAGES ====\n")
        if images:
            for img in images:
                f.write(f"{img}\n")
        else:
            f.write("No images extracted.\n")

        f.write("\n==== OCR TEXT FROM IMAGES ====\n")
        f.write(image_ocr_text + "\n")

    print(f"   [✔] Done → {txt_filename}")
