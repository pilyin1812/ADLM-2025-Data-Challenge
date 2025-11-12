import pymupdf4llm
from pathlib import Path
import re
import traceback

# Strict section header pattern
section_header_pattern = re.compile(
    r"^(?:\*\*)?(?:[A-Z]|\d{1,2})\.\s+[A-Z][A-Z\s\-]{2,}$"
)

def extract_title_block(md_text: str, filename: str) -> str:
    lines = md_text.splitlines()
    title_lines = []
    in_title = False

    for line in lines:
        stripped = line.strip().strip("*")
        lowered = stripped.lower()

        if not in_title:
            if lowered.startswith("standard operating procedure") or lowered.startswith("protocol for the analytical phase of"):
                in_title = True

        if in_title:
            if section_header_pattern.match(stripped):
                break
            title_lines.append(stripped)

    if title_lines:
        title_block = "\n".join(title_lines).strip()
        tagged = f"<!-- section: title | source: {filename} -->\n# {title_lines[0]}\n\n" + title_block
        remaining = md_text.replace("\n".join(title_lines), "").strip()
        return tagged + "\n\n" + remaining

    return md_text

def detect_letter_number_sections(md_text: str) -> str:
    pattern = re.compile(
        r"^(?:\*\*)?(?:[A-Z]|\d{1,2})\.\s+[A-Z][A-Z\s\-]{2,}$",
        re.MULTILINE
    )

    def replacer(match):
        title = match.group(0).strip()
        tag = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        return f"<!-- section: {tag} -->\n## {title}"

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
        r"\b\d+(\.\d+)?\s*(°C|°F|%|mg/mL|µL|mL|g|kg|mm|cm|nm|μg|ng|mol/L)\b"
    )

    leading_strip = re.compile(
        r"^(?:\*\*|\d+[\.\)\-:]|\b[A-Z][\.\)\-:]|\b[MCDXLIV]+\b[\.\)\-:]?)\s*",
        re.I,
    )
    inline_strip = re.compile(
        r"(?:[\*\◦\•]+|\.{1,}(?=\s{3,})|\d+\.(?=\n{2,})|^\s*\d+\.\s*$)", re.M
    )

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            continue

        if scientific_units.search(stripped):
            formatted_lines.append(stripped)
            continue

        stripped = stripped.replace("**", "")
        if section_header_pattern.match(stripped):
            clean_header = leading_strip.sub("", stripped)
            formatted_lines.append(f"<!-- section: {clean_header.lower()} | source: {filename} -->")
            formatted_lines.append("## " + clean_header)
        else:
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
        if not md_text.strip():
            print(f"⚠️ Empty Markdown output for {pdf_path}")
            return

        filename = pdf_path.name
        no_strikethrough = remove_strikethrough(md_text)
        with_title = extract_title_block(no_strikethrough, filename)
        md_with_headers = clean_markdown_headers(with_title, filename)
        tagged_sections = detect_letter_number_sections(md_with_headers)

        out_path.write_text(tagged_sections, encoding="utf-8")
        print(f"✅ Saved: {out_path}")
    except Exception as e:
        print(f"❌ Failed to process {pdf_path}: {e}")
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