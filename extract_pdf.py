import fitz  # PyMuPDF
import sys
import os
import shutil
import pymupdf4llm # For extracting markdown natively with math/tables

def extract_pdf(pdf_path, output_dir):
    """
    Extracts text as markdown from a PDF file using pymupdf4llm.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        return

    # Create output directory for this PDF
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_out_dir = os.path.join(output_dir, base_name)
    os.makedirs(doc_out_dir, exist_ok=True)
    
    images_dir = os.path.join(doc_out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    text_output_path = os.path.join(doc_out_dir, f"{base_name}_text.md")
    
    print(f"Opening {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Failed to open PDF: {e}")
        return

    total_pages = len(doc)
    
    with open(text_output_path, "w", encoding="utf-8") as text_file:
        text_file.write(f"# Extracted Content: {base_name}\n\n")
        
        for page_num in range(total_pages):
            print(f"Processing page {page_num + 1}/{total_pages}...")
            
            # Use pymupdf4llm to extract high quality markdown (preserves math, tables)
            # and automatically extracts images!
            md_text = pymupdf4llm.to_markdown(
                doc,
                pages=[page_num],
                write_images=True,
                image_path=images_dir,
                image_format="png"
            )
            
            # FAIL-SAFE: Take an actual snapshot image of the full page in case math breaks
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=150)
            snapshot_filename = f"snapshot_page_{page_num + 1}.png"
            snapshot_path = os.path.join(images_dir, snapshot_filename)
            pix.save(snapshot_path)

            text_file.write(f"## Page {page_num + 1}\n\n")
            
            # Embed the fail-safe visual snapshot directly into the markdown FIRST
            text_file.write(f"### Visual Page Snapshot:\n")
            text_file.write(f"![Page {page_num + 1} Snapshot](images/{snapshot_filename})\n\n")
            
            if md_text.strip():
                text_file.write("### Extracted Markdown (pymupdf4llm):\n")
                text_file.write(md_text)
                text_file.write("\n\n")

def extract_single_image(img_path, output_dir):
    print("Extract single image not implemented")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_pdf.py <path_to_file> <output_dir>")
    else:
        file_path = sys.argv[1]
        out_dir = sys.argv[2]
            
        ext = file_path.lower().split('.')[-1]
        if ext == 'pdf':
            extract_pdf(file_path, out_dir)
        elif ext in ['png', 'jpg', 'jpeg']:
            extract_single_image(file_path, out_dir)
        else:
            print(f"Unsupported file type: {ext}")
