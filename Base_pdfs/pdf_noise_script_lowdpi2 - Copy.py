import os
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import io
from pathlib import Path

def add_scanning_artifacts(image, noise_level=0.3):
    """Add realistic scanning artifacts to an image"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Add subtle noise
    noise = np.random.normal(0, noise_level * 5, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # Convert back to PIL
    result = Image.fromarray(noisy_img)
    
    # Add paper grain texture
    grain_intensity = random.uniform(0.8, 1.5)  # Random grain amount
    grain = np.random.normal(0, grain_intensity, img_array.shape)
    
    # Apply grain with reduced intensity in darker areas (more realistic)
    img_brightness = np.mean(img_array, axis=2, keepdims=True) if len(img_array.shape) == 3 else img_array
    grain_mask = img_brightness / 255.0  # Normalize to 0-1
    grain = grain * grain_mask  # Less grain in dark areas
    
    grained_img = np.clip(img_array + grain, 0, 255).astype(np.uint8)
    result = Image.fromarray(grained_img)
    
    # Add very subtle salt and pepper noise (dust/specs)
    if random.random() < 0.4:  # 40% chance
        dust_pixels = int(img_array.size * 0.0001)  # 0.01% of pixels
        for _ in range(dust_pixels):
            if len(img_array.shape) == 3:
                y, x = random.randint(0, img_array.shape[0]-1), random.randint(0, img_array.shape[1]-1)
                if random.random() < 0.5:
                    grained_img[y, x] = [max(0, grained_img[y, x, c] - random.randint(10, 30)) for c in range(3)]
                else:
                    grained_img[y, x] = [min(255, grained_img[y, x, c] + random.randint(10, 30)) for c in range(3)]
        result = Image.fromarray(grained_img)
    
    # Add slight blur (scanner motion blur)
    if random.random() < 0.3:  # 30% chance
        blur_radius = random.uniform(0.1, 0.3)
        result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Adjust contrast slightly (scanner inconsistency)
    contrast_factor = random.uniform(0.95, 1.05)
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(contrast_factor)
    
    # Adjust brightness slightly
    brightness_factor = random.uniform(0.97, 1.03)
    enhancer = ImageEnhance.Brightness(result)
    result = enhancer.enhance(brightness_factor)
    
    return result

def add_artifacts_to_pdf(input_path, output_path, 
                        rotation_range=(-1.5, 1.5), 
                        noise_level=0.3,
                        dpi=200):
    """Add scanning artifacts to PDF while keeping it OCR-friendly"""
    try:
        doc = fitz.open(input_path)
        new_doc = fitz.open()
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Random rotation within specified range
            rotation = random.uniform(rotation_range[0], rotation_range[1])
            
            # Get page center for rotation
            rect = page.rect
            center_x = (rect.x0 + rect.x1) / 2
            center_y = (rect.y0 + rect.y1) / 2
            center = fitz.Point(center_x, center_y)
            
            # Create transformation matrix for rotation
            mat = fitz.Matrix(1, 1).prerotate(rotation)
            
            # Render page as high-quality image
            # Higher DPI maintains quality for OCR
            zoom = dpi / 72.0  # 72 DPI is default
            render_mat = fitz.Matrix(zoom, zoom) * mat
            pix = page.get_pixmap(matrix=render_mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Add scanning artifacts
            processed_image = add_scanning_artifacts(image, noise_level)
            
            # Convert back to bytes with compression
            img_buffer = io.BytesIO()
            # Use JPEG with high quality for much smaller file sizes
            if processed_image.mode == 'RGBA':
                # Convert RGBA to RGB for JPEG
                rgb_image = Image.new('RGB', processed_image.size, (255, 255, 255))
                rgb_image.paste(processed_image, mask=processed_image.split()[-1])
                processed_image = rgb_image
            
            processed_image.save(img_buffer, format='JPEG', quality=85, optimize=True)
            img_buffer.seek(0)
            
            # Calculate new page size to accommodate rotation
            original_rect = page.rect
            if abs(rotation) > 0.1:
                # Expand page size slightly to accommodate rotation
                margin = max(original_rect.x1 - original_rect.x0, original_rect.y1 - original_rect.y0) * 0.05
                new_width = (original_rect.x1 - original_rect.x0) + margin
                new_height = (original_rect.y1 - original_rect.y0) + margin
            else:
                new_width = original_rect.x1 - original_rect.x0
                new_height = original_rect.y1 - original_rect.y0
            
            # Create new page
            new_page = new_doc.new_page(width=new_width, height=new_height)
            
            # Insert processed image
            img_rect = fitz.Rect(0, 0, new_width, new_height)
            new_page.insert_image(img_rect, stream=img_buffer.getvalue())
        
        # Save with optimization for smaller file size
        new_doc.save(output_path, 
                    garbage=4,  # Remove unused objects
                    deflate=True,  # Compress
                    clean=True)  # Clean up document structure
        new_doc.close()
        doc.close()
        
        print(f"✓ Processed: {os.path.basename(input_path)}")
        return True
    
    except Exception as e:
        print(f"✗ Error processing {os.path.basename(input_path)}: {str(e)}")
        return False

def main():
    # Configuration - uses current directory
    input_dir = os.getcwd()  # Current directory where script is located
    output_dir = os.path.join(input_dir, "scanning_artifacts")
    
    # Artifact settings (adjust these to control realism)
    settings = {
        'rotation_range': (-1.0, 1.0),  # degrees, realistic scanner skew
        'noise_level': 0.3,  # Increased from 0.2 for more graininess
        'dpi': 120  # Reduced to 120 - minimum for good OCR, much smaller files
    }
    
    print("PDF Scanning Artifact Generator")
    print(f"Resolution: {settings['dpi']} DPI")
    print(f"Rotation range: {settings['rotation_range']} degrees")
    print(f"Noise level: {settings['noise_level']}")
    print(f"Image format: JPEG")
    print(f"JPEG quality: 85% (lossy compression)")
    print(f"Color space: RGB (24-bit)")
    print(f"Compression: Deflate + document optimization")
    print(f"Artifacts: Paper grain, salt-pepper noise, motion blur")
    print("-" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find PDF files
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files. Processing...")
    
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        input_path = str(pdf_file)
        output_filename = f"{pdf_file.stem}_scanned.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        if add_artifacts_to_pdf(input_path, output_path, **settings):
            successful += 1
        else:
            failed += 1
    
    print(f"\nCompleted! {successful} successful, {failed} failed.")
    print(f"Output files saved to: {output_dir}")
    print("\nFiles now have realistic scanning artifacts while remaining OCR-friendly!")

if __name__ == "__main__":
    main()