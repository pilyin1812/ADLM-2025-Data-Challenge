from pathlib import Path
import shutil
#Sorts FDA PDFs into Review and Raw folders
#pdfs with _REVIEW.pdf can be converted to markdown with procedures-to-markdown.py
#pdfs without _REVIEW.pdf can be converted to markdown with OCR_test.py
# --- Config ---
FDA_ROOT = Path("LabDocs/FDA")
REVIEW_DIR = Path("Sorted/FDA/Review_PDFs")
RAW_DIR = Path("Sorted/FDA/PDFs")

def sort_fda_pdfs():
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    review_count = 0
    raw_count = 0

    for pdf in FDA_ROOT.glob("**/*.pdf"):
        if "_REVIEW.pdf" in pdf.name:
            target = REVIEW_DIR / pdf.name
            shutil.move(pdf, target)
            review_count += 1
        else:
            target = RAW_DIR / pdf.name
            shutil.move(pdf, target)
            raw_count += 1

    print(f"✅ Moved {review_count} REVIEW PDFs to {REVIEW_DIR}")
    print(f"✅ Moved {raw_count} RAW PDFs to {RAW_DIR}")

if __name__ == "__main__":
    sort_fda_pdfs()