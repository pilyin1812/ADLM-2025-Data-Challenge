import pymupdf4llm
from pathlib import Path
import re
import traceback


section_header_pattern = re.compile(
    r"^(?:\*\*)?(?:[A-Z]|\d{1,2})\.\s+[A-Z][\w\s\-/():]*:?$", re.MULTILINE
)

subsection_pattern = re.compile(r"^(\d{1,2})\.\s+([A-Z][\w\s\-/():]*):?$")

def extract_identifier_block(md_text: str, filename: str) -> tuple[str, str | None]:
    lines = md_text.splitlines()
    identifier = None
    capture_next = False

    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()

        if "proprietary and established names" in lowered or "manufacturer and instrument name" in lowered:
            capture_next = True
            continue

        if capture_next and stripped:
            identifier = stripped
            break

    if identifier:
        tag = re.sub(r"[^a-z0-9]+", "_", identifier.lower()).strip("_")
        tagged = f"<!-- section: {tag} | source: {filename} -->\n# {identifier}\n\n"
        return tagged + md_text, tag

    return md_text, None

def detect_letter_number_sections(md_text: str, filename: str, identifier_tag: str | None = None) -> str:
    pattern = section_header_pattern

    def replacer(match):
        title = match.group(0).strip()
        section_tag = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        full_tag = f"<!-- section: {section_tag} | source: {filename}"
        if identifier_tag:
            full_tag += f" | {identifier_tag}"
        full_tag += " -->"
        return f"{full_tag}\n## {title}"

    return pattern.sub(replacer, md_text)

def remove_strikethrough(md_text: str) -> str:
    return re.sub(r"~~(.*?)~~", r"\1", md_text, flags=re.DOTALL)

def normalize_ranges(text: str) -> str:
    text = re.sub(r"(\d+)\s*[-–—]\s*(\d+)(\s*[°%]?[CF]?)", r"\1–\2\3", text)
    text = text.replace("degrees Celsius", "°C").replace("degrees Fahrenheit", "°F")
    return text

def clean_markdown_headers(text: str, filename: str) -> str:
    text = normalize_ranges(text)

    formatted_lines = []
    scientific_units = re.compile(
        r"\b\d+(\.\d+)?\s*(°C|°F|%|mg/mL|µL|mL|g|kg|mm|cm|nm|μg|ng|mol/L|µg/mL|ug/mL)\b"
    )

    leading_strip = re.compile(
        r"^(?:\*\*|\d+[\.\)\-:]|\b[A-Z][\.\)\-:]|\b[MCDXLIV]+\b[\.\)\-:]?)\s*",
        re.I,
    )
    inline_strip = re.compile(
        r"(?:[\*\◦\•]+|\.{1,}(?=\s{3,})|\d+\.(?=\n{2,})|^\s*\d+\.\s*$)", re.M
    )

    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") or re.match(r"^\s*<table>", stripped):
            in_table = True
        elif in_table and not stripped:
            in_table = False

        if in_table:
            formatted_lines.append(line)
            continue

        if not stripped:
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            continue

        if scientific_units.search(stripped):
            formatted_lines.append(stripped)
            continue

        stripped = stripped.replace("**", "")
        clean_line = inline_strip.sub("", stripped)
        formatted_lines.append(clean_line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(formatted_lines))

def convert_one_pdf(pdf_path: Path, pdf_dir: Path, output_dir: Path):
    rel = pdf_path.relative_to(pdf_dir).with_suffix(".md")
    out_path = output_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Converting: {pdf_path}")
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        md_text, identifier_tag = extract_identifier_block(md_text, pdf_path.name)
        md_text = detect_letter_number_sections(md_text, pdf_path.name, identifier_tag)
        md_text = clean_markdown_headers(md_text, pdf_path.name)
        out_path.write_text(md_text, encoding="utf-8")
        print(f"✅ Saved: {out_path}")
    except Exception as e:
        print(f"❌ Failed to process {pdf_path.name}: {e}")
        traceback.print_exc()

def batch_convert_pdfs(pdf_dir: str, output_dir: str):
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = list(pdf_dir.rglob("*.pdf"))

    print(f"📁 Found {len(pdf_paths)} PDFs in {pdf_dir}")
    if not pdf_paths:
        print("⚠️ No PDFs found. Check your path or file extensions.")
        return

    for pdf_path in pdf_paths:
        convert_one_pdf(pdf_path, pdf_dir, output_dir)

# Run the batch conversion
if __name__ == "__main__":
    batch_convert_pdfs(
        pdf_dir="Sorted_FDA/PDFs",
        output_dir="Markdown-Output/FDA"
    )