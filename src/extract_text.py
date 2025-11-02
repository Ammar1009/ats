# src/extract_text.py
import pdfplumber

def extract_text_from_pdf(file_obj):
    """
    file_obj: file-like (BytesIO from Streamlit or open path)
    returns: extracted text string
    """
    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text