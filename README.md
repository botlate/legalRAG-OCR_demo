# legalRAG-OCR
## OCR Pipeline for Legal Document Processing

### Overview
Working OCR module for a demonstration RAG (Retrieval-Augmented Generation) pipeline for querying small sets of legal pleadings with high precision/low hallucinations. This module converts scanned court documents into machine-readable text using smart layout recognition and GPU-accelerated OCR.

### Technical Implementation
- DocTR deep learning OCR engine
- Docker containerization for environment consistency
- GPU acceleration using CUDA and PyTorch
- Layout analysis with PaddleOCR for document type detection (CPU)
- Batch processing for multiple images

### Technical Stack
- Python 3.10+
- Docker with NVIDIA CUDA support
- PyTorch with GPU acceleration
- DocTR for OCR
- pdf2image for image extraction
- OpenCV for image processing
- PaddleOCR for layout analysis & cropping

### Test Dataset
Processed 19 documents from multiple rounds of demurrer proceedings in _Glenn Mahler, et al. v. Judicial Council of California_ (San Francisco Sup. Ct. case no. CGC-19-575842) (filed May 9, 2019):
- Document types: complaints, demurrers, oppositions, replies, orders
- Total pages: ~380
- Mixed formats: line-numbered pleadings, standard forms, various exhibits

### Architecture

#### Processing Pipeline
```
PDF Documents → Image Extraction → Layout Analysis → Smart Cropping → GPU OCR → Text Files
```

#### File Organization
| Folder | Purpose | Script |
|--------|---------|--------|
| `Base_pdfs/` | Input PDF documents | - |
| `Extracted_images/` | Individual page images (300 DPI) | `extract_images.py` |
| `Cropped_images/` | Deskewed & cropped images | `layout_aware_crop.py` |
| `OCR_output/` | Final text output files | `run_doctr.py` (in Docker) |

#### Docker Structure
| Component | Purpose |
|-----------|---------|
| `Dockerfile` | CUDA 12.2 + PyTorch + DocTR environment |
| `run_doctr.py` | OCR script running inside container |

### Layout-Aware Processing
The pipeline differentiates between:
- Court pleadings with line numbers (removes left margin - high accuracy)
- Non-pleadings without line numbers (leaves alone - mixed accuracy)
- Uses confidence scoring to determine document type and crop boundaries
- Currently limited to cropping out line numbers, more advanced layout detection needed

### Observed Performance
- Processing speed: 1-1.5 seconds per page using RTX 4070
- Batch processing implemented to maximize GPU utilization
- Automatic line number detection and removal for court pleadings largely successful
- Some manual re-cropping needed
- Current speed is too slow. May be because OCR model reloads model or content into VRAM each time. Larger document sets may have faster rates.

### Demo
See the `/examples` folder for:
- Sample input (scanned pleading with line numbers)
- Processed output (clean, searchable text)
- Performance benchmarks

### Future Steps: The Full RAG Pipeline
This OCR module is the foundation for:
1. **legalRAG-chunking**: Intelligent document segmentation and addition of metadata to aid embedding/query
2. **legalRAG-embedding**: Vector embeddings intended to maintain exact language  
3. **legalRAG-vectorDB**: Storage and retrieval
4. **legalRAG-query**: Natural language search across case law

### Project Context
This OCR module is designed as the first step in building a complete RAG pipeline for small sets of legal documents to be run on local LLM. The goal is to enable very high accuracy of semantic search querying of pleadings.

---
*Demonstration of OCR component for legal document processing pipeline*
