# Known Issues and Areas for Improvement

## Current Limitations

### 1. Mixed Quality OCR Results
- **Issue**: Some documents produce poor OCR quality despite clean scans
- **Needs Investigation**: Why DocTR performs poorly on certain layouts

### 2. OCR Performance
- **Issue**: Processing speed of 1-1.5 seconds per page is too slow for large document sets
- **Suspected Cause**: Model may be reloading weights to VRAM
- **Impact**: 380 pages takes ~8 minutes
- **Potential Solutions**: 
  - May not be doing true batch processing
  - Profile VRAM usage during execution

### 3a. Layout Detection Accuracy
- **Issue**: Non-pleading documents (exhibits, forms) sometimes get incorrectly cropped
- **Example**: Cover pages, signature pages, and exhibits may have text cut off
- **Current Workaround**: Manual review and re-cropping needed
- **Potential Solutions**:
  - Legal document specific layout
  - Add page-type detection (first page, signature page, exhibit)
  - Better confidence threshold for skipping uncertain pages

### 3b. Missing Document Structure
- **Issue**: OCR output is plain text, losing important formatting
- **Lost Information**:
  - Paragraph boundaries
  - Headers and section numbers
  - Page numbers
  - Footnote references
- **Impact**: Will make chunking  difficult

