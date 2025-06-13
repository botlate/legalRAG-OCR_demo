import os
import cv2
import numpy as np

# Try PaddleOCR approach first (more reliable)
try:
    from paddleocr import PaddleOCR
    USE_PADDLE = True
    print("Using PaddleOCR for layout analysis")
    ocr_model = PaddleOCR(use_angle_cls=False, lang='en', show_log=False, 
                         use_space_char=True, layout=True)
except ImportError:
    print("PaddleOCR not found, install with: pip install paddleocr")
    USE_PADDLE = False

# === Configuration ===
input_dir = r"C:\AI-dem\Extracted_images"
output_dir = r"C:\AI-dem\Cropped_images"
os.makedirs(output_dir, exist_ok=True)

def deskew(image):
    """Same deskew function as before"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)

    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            angle_deg = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
            angles.append(angle_deg)
        if angles:
            angle = np.median([a for a in angles if -45 < a < 45])

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def detect_line_numbers_with_layout(image):
    """
    Use PaddleOCR layout analysis to detect if this looks like a pleading with line numbers
    Returns: (has_line_numbers: bool, confidence: float)
    """
    if not USE_PADDLE:
        return fallback_detection(image)
    
    try:
        # PaddleOCR returns layout regions and text
        result = ocr_model.ocr(image, cls=False)
        
        if not result or not result[0]:
            return False, 0.0
        
        # Extract text block coordinates and OCR confidence
        text_regions = []
        ocr_confidences = []
        
        for line in result[0]:
            if line and len(line) >= 2:
                bbox = line[0]  # Bounding box coordinates
                text_info = line[1]  # Text and confidence
                
                # Get leftmost x coordinate
                left_x = min(point[0] for point in bbox)
                text_regions.append(left_x)
                
                # Get OCR confidence
                if isinstance(text_info, tuple) and len(text_info) >= 2:
                    ocr_confidences.append(text_info[1])
                else:
                    ocr_confidences.append(0.5)  # Default confidence
        
        if len(text_regions) < 3:
            return False, 0.0
        
        # Calculate layout confidence based on text positioning
        image_width = image.shape[1]
        indent_threshold = image_width * 0.08
        indented_count = sum(1 for x in text_regions if x > indent_threshold)
        indent_ratio = indented_count / len(text_regions)
        
        # Calculate overall confidence
        avg_ocr_confidence = np.mean(ocr_confidences) if ocr_confidences else 0.0
        
        # Layout confidence: how consistently indented the text is
        layout_confidence = min(1.0, indent_ratio * 1.5)  # Scale up slightly
        
        # Combined confidence (weighted average)
        combined_confidence = (layout_confidence * 0.7) + (avg_ocr_confidence * 0.3)
        
        # Decision threshold
        has_line_numbers = indent_ratio >= 0.6
        
        return has_line_numbers, combined_confidence
        
    except Exception as e:
        print(f"PaddleOCR layout analysis failed: {e}")
        return fallback_detection(image)

def find_text_start_x(image):
    """
    Use layout analysis to find where main text actually starts
    Returns: (crop_x: int, confidence: float)
    """
    if not USE_PADDLE:
        return fallback_crop_x(image)
    
    try:
        result = ocr_model.ocr(image, cls=False)
        
        if not result or not result[0]:
            return 0, 0.0
        
        # Find leftmost edges of text blocks with their confidences
        left_edges = []
        confidences = []
        
        for line in result[0]:
            if line and len(line) >= 2:
                bbox = line[0]
                text_info = line[1]
                
                left_x = min(point[0] for point in bbox)
                left_edges.append(left_x)
                
                # Get confidence
                if isinstance(text_info, tuple) and len(text_info) >= 2:
                    confidences.append(text_info[1])
                else:
                    confidences.append(0.5)
        
        if not left_edges:
            return 0, 0.0
        
        # Find the most common left edge (main text column)
        left_edges_array = np.array(left_edges)
        confidences_array = np.array(confidences)
        
        # Weight the edges by their OCR confidence
        weighted_edges = left_edges_array * confidences_array
        median_left = np.median(weighted_edges) / np.median(confidences_array)
        
        # Calculate crop confidence based on consistency of left edges
        edge_std = np.std(left_edges_array)
        consistency_confidence = max(0.0, 1.0 - (edge_std / 100))  # Lower std = higher confidence
        avg_ocr_confidence = np.mean(confidences_array)
        
        crop_confidence = (consistency_confidence * 0.6) + (avg_ocr_confidence * 0.4)
        
        # Add small padding
        crop_x = max(0, int(median_left - 15))
        return crop_x, crop_confidence
        
    except Exception as e:
        print(f"PaddleOCR crop detection failed: {e}")
        return fallback_crop_x(image)

def fallback_detection(image):
    """Simple fallback detection method with confidence"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    projection = np.sum(binary, axis=0)
    width = image.shape[1]
    check_width = int(width * 0.15)
    left_projection = projection[:check_width]
    
    if np.max(left_projection) == 0:
        return False, 0.0
    
    smoothed = np.convolve(left_projection, np.ones(10), mode='same')
    norm = smoothed / np.max(smoothed)
    
    peak_found = False
    peak_value = 0
    drop_ratio = 0
    
    for x in range(30, min(check_width, int(width * 0.2))):
        if norm[x] > 0.4:
            peak_found = True
            peak_value = max(peak_value, norm[x])
        elif peak_found and norm[x] < peak_value * 0.3:
            drop_ratio = (peak_value - norm[x]) / peak_value
            confidence = min(0.8, drop_ratio)  # Max 0.8 for fallback method
            return True, confidence
    
    return False, 0.0

def fallback_crop_x(image):
    """Original crop detection as fallback with confidence"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    projection = np.sum(binary, axis=0)
    width = image.shape[1]
    max_crop_width = int(width * 0.25)
    smoothed = np.convolve(projection, np.ones(10), mode='same')
    
    if np.max(smoothed) == 0:
        return 0, 0.0
    
    norm = smoothed / np.max(smoothed)
    threshold = 0.2
    peak_found = False
    
    for x in range(30, max_crop_width):
        if norm[x] > 0.4:
            peak_found = True
        elif peak_found and norm[x] < threshold:
            # Calculate confidence based on how clear the drop is
            peak_val = np.max(norm[max(0, x-30):x])
            drop_val = norm[x]
            confidence = min(0.7, (peak_val - drop_val) / peak_val)
            return x, confidence
    
    return max_crop_width, 0.3  # Low confidence fallback

# === Process all images ===
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
        continue

    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading: {img_path}")
        continue

    # Always deskew
    deskewed = deskew(img)
    
    try:
        # Use layout analysis to detect document type
        has_line_numbers, detection_confidence = detect_line_numbers_with_layout(deskewed)
        
        if has_line_numbers:
            # Use layout analysis to find precise crop point
            crop_x, crop_confidence = find_text_start_x(deskewed)
            cropped = deskewed[:, crop_x:]
            print(f"Layout analysis: pleading detected (conf: {detection_confidence:.2f}) - cropped at x={crop_x} (conf: {crop_confidence:.2f}): {fname}")
        else:
            cropped = deskewed
            print(f"Layout analysis: no line numbers detected (conf: {detection_confidence:.2f}): {fname}")
            
    except Exception as e:
        print(f"Layout analysis failed for {fname}, using fallback: {e}")
        # Fallback to simple detection
        has_line_numbers, detection_confidence = fallback_detection(deskewed)
        if has_line_numbers:
            crop_x, crop_confidence = fallback_crop_x(deskewed)
            cropped = deskewed[:, crop_x:]
            print(f"Fallback: cropped at x={crop_x} (conf: {crop_confidence:.2f}): {fname}")
        else:
            cropped = deskewed
            print(f"Fallback: no cropping (conf: {detection_confidence:.2f}): {fname}")

    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, cropped)
    print(f"Saved: {out_path}")
