import fitz 
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        full_text += f"\n--- Page {page_num} ---\n{text}"
    return full_text

def save_text_to_file(text: str, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

def process_pdf(pdf_path: str, output_dir: str = "output_text"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.txt")
    text = extract_text_from_pdf(pdf_path)
    save_text_to_file(text, output_path)
    return output_path