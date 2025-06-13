import os

base_dir = r"C:\AI-dem"

folders_to_create = [
    "Base_pdfs",
    "Extracted_images",
    "Cropped_images",
    "OCR_output",
    "OCR_output_batch",
    os.path.join("doctr_gpu_batchtest")
]

for folder in folders_to_create:
    full_path = os.path.join(base_dir, folder)
    os.makedirs(full_path, exist_ok=True)
    print(f"Created: {full_path}")
