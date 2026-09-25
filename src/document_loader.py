from pypdf import PdfReader

def load_pdf_text(pdf_path):
    """PDF se saara text extract karta hai, page by page."""
    reader = PdfReader(pdf_path)
    all_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        all_text += page_text + "\n"
    return all_text