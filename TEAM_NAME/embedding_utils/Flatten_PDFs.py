import subprocess
import shutil
from pathlib import Path

def flatten_pdf(input_path: Path, output_path: Path, failed_dir: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ghostscript_path = r"C:\Program Files\gs\gs10.06.0\bin\gswin64c.exe"  # Adjust version if needed

        subprocess.run([
            ghostscript_path,
            "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
            "-dPDFSETTINGS=/prepress",
            f"-sOutputFile={str(output_path)}",
            str(input_path)
        ], check=True)
        print(f"✅ Flattened: {input_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to flatten {input_path.name}: {e}")
        failed_path = failed_dir / input_path.name
        failed_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, failed_path)
        print(f"📁 Moved to failed folder: {failed_path}")

def batch_flatten_pdfs(source_dir: str, output_dir: str, failed_dir: str):
    source = Path(source_dir)
    output = Path(output_dir)
    failed = Path(failed_dir)
    pdfs = list(source.rglob("*.pdf"))

    print(f"📁 Found {len(pdfs)} PDFs to flatten in {source_dir}")
    for pdf in pdfs:
        rel_path = pdf.relative_to(source)
        out_path = output / rel_path
        flatten_pdf(pdf, out_path, failed)

if __name__ == "__main__":
    batch_flatten_pdfs(
        source_dir="Sorted/FDA/Raw_PDFs",
        output_dir="Flattened/FDA/Raw_PDFs",
        failed_dir="Flattened/Failed"
    )