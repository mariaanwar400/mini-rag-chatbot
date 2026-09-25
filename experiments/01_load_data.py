from pypdf import PdfReader

# PDF file ka path
pdf_path = "data/document.pdf"

# PDF reader object banayein
reader = PdfReader(pdf_path)

# Total pages check karein
print(f"Total pages: {len(reader.pages)}")

# Har page se text extract karein aur ek list mein store karein
all_text = ""
for page_number, page in enumerate(reader.pages):
    page_text = page.extract_text()
    all_text += page_text + "\n"

# Sanity check
print(f"Total characters extracted: {len(all_text)}")
print("\n--- First 500 characters (preview) ---\n")
print(all_text[:500])