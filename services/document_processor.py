from pypdf import PdfReader
import os
def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
def read_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
def read_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text
def extract_text(file_path):

    ext = os.path.splitext(file_path)[1]

    if ext == ".txt":
        return read_txt(file_path)

    elif ext == ".md":
        return read_md(file_path)

    elif ext == ".pdf":
        return read_pdf(file_path)

    else:
        raise ValueError("Unsupported file type")
def chunk_text(
        text,
        chunk_size=1000,
        overlap=200
):
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks