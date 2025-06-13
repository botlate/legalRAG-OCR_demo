import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch

device = torch.device("cuda:1")

input_dir = "/workspace/input_images"
output_dir = "/workspace/output"
os.makedirs(output_dir, exist_ok=True)

# Load model once
model = ocr_predictor(pretrained=True, assume_straight_pages=True).to(device)

# Collect all image paths
image_paths = [
    os.path.join(input_dir, fname)
    for fname in sorted(os.listdir(input_dir))
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff'))
]

# Run OCR on all images at once
doc = DocumentFile.from_images(image_paths)
result = model(doc)

# Write output per page
for i, page in enumerate(result.pages):
    fname = os.path.basename(image_paths[i])
    out_path = os.path.join(output_dir, fname + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page.render())
