import os
from pdf2image import convert_from_path

base_dir = r"C:\AI-dem"
pdf_dir = os.path.join(base_dir, "Base_pdfs")
out_dir = os.path.join(base_dir, "Extracted_images")
os.makedirs(out_dir, exist_ok=True)

for fname in os.listdir(pdf_dir):
    if not fname.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_dir, fname)
    base_name = os.path.splitext(fname)[0]

    images = convert_from_path(pdf_path, dpi=300)
    for i, img in enumerate(images):
        out_path = os.path.join(out_dir, f"{base_name}_page_{i+1:03}.jpg")
        img.save(out_path, "JPEG")
        print(f"Saved {out_path}")
