import sys
from typing import List, Tuple, Dict, Optional
import pypdfium2 as pdfium
from statistics import median
from pathlib import Path
from typing import List
from PIL import Image
import pytesseract


# If Tesseract isn't on PATH, paste the path to your Tesseract.exe here:

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\barry\OneDrive\Desktop\Tessaract\tesseract.exe"




# --- Config ---
INPUT_DIR = "Flattened/FDA/Review_PDFs"
OUTPUT_DIR = "Markdown-Output/FDA/OCR"
DPI = 300
LANG = "eng"

# --- OCR Pipeline ---
def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    pdf = pdfium.PdfDocument(pdf_path)
    scale = dpi / 72.0
    pages = []
    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=scale).to_pil().convert("RGB")
        pages.append(pil)
    return pages

def run_ocr_on_image(image: Image.Image, lang: str = "eng") -> str:
    import pytesseract
    return pytesseract.image_to_string(image, lang=lang)

def process_pdf(pdf_path: Path, output_dir: Path):
    print(f"📄 Processing: {pdf_path.name}")
    images = pdf_to_images(str(pdf_path), dpi=DPI)
    markdown_lines = [f"# OCR Output for {pdf_path.name}\n"]

    for i, image in enumerate(images):
        text = run_ocr_on_image(image, lang=LANG)
        markdown_lines.append(f"\n<!-- page: {i + 1} -->\n{text.strip()}")

    output_path = output_dir / (pdf_path.stem + ".md")
    output_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")
    print(f"✅ Saved: {output_path.name}")


def detect_orientation(img: Image.Image) -> Image.Image:
    # Auto-rotate a page image using Tesseract's orientation detection.
    try:
        osd = pytesseract.image_to_osd(img)
        angle = 0
        for line in osd.splitlines():
            if "Rotate:" in line:
                angle = int(line.split(":")[1].strip())
                break
        if angle:
            return img.rotate(-angle, expand=True)
    except Exception:
        pass
    return img


def tesseract_words(img: Image.Image, lang: str = "eng") -> List[Dict]:
    """
    Extract word-level boxes using Output.DICT to avoid pandas dependency.
    Use psm 6 ("Assume a single uniform block of text") – tends to work
    much better for tables than psm 4 on scans.
    """
    data = pytesseract.image_to_data(
        img,
        lang=lang,
        output_type=pytesseract.Output.DICT,
        config="--oem 1 --psm 6",
    )
    n = len(data["text"])
    words: List[Dict] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data.get("conf", [-1]*n)[i])
        except Exception:
            conf = -1.0
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        words.append(
            {"text": txt, "conf": conf, "x1": x, "y1": y, "x2": x + w, "y2": y + h}
        )
    return words


def group_lines(words: List[Dict], y_tol_ratio: float = 0.6) -> List[List[Dict]]:
    if not words:
        return []
    heights = [w["y2"] - w["y1"] for w in words]
    heights.sort()
    med_h = heights[len(heights) // 2] or 1
    y_tol = max(4, int(med_h * y_tol_ratio))

    words_sorted = sorted(words, key=lambda w: (w["y1"], w["x1"]))
    lines: List[List[Dict]] = []
    current: List[Dict] = []

    def same_line(a, b) -> bool:
        ay = (a["y1"] + a["y2"]) // 2
        by = (b["y1"] + b["y2"]) // 2
        return abs(ay - by) <= y_tol

    for w in words_sorted:
        if not current:
            current = [w]
            continue
        if same_line(current[-1], w):
            current.append(w)
        else:
            lines.append(sorted(current, key=lambda x: x["x1"]))
            current = [w]
    if current:
        lines.append(sorted(current, key=lambda x: x["x1"]))
    return lines


def _bin_xs(xs: List[int], tol: int) -> List[float]:
    """
    Greedy 1D clustering for x-centers; returns cluster centers.
    """
    if not xs:
        return []
    xs_sorted = sorted(xs)
    centers: List[float] = []
    counts: List[int] = []
    for x in xs_sorted:
        if not centers or abs(x - centers[-1]) > tol:
            centers.append(float(x))
            counts.append(1)
        else:
            # update running average
            k = len(centers) - 1
            centers[k] = (centers[k] * counts[k] + x) / (counts[k] + 1)
            counts[k] += 1
    return centers


def detect_table_blocks(lines: List[List[Dict]]) -> List[Tuple[int, int, List[float]]]:
    """
    Identify blocks of lines that appear to form tables.
    Returns (start_idx, end_idx, column_centers) so we can reuse bins during rendering.
    """
    tables: List[Tuple[int, int, List[float]]] = []
    i = 0
    while i < len(lines):
        start = i
        col_bins: Optional[List[float]] = None
        run = 0
        # track a running estimate of column centers (median is robust)
        est_bins_history: List[List[float]] = []

        while i < len(lines):
            xs = [(w["x1"] + w["x2"]) // 2 for w in lines[i]]
            # If too few items on a line, unlikely to be a row of a table
            if len(xs) < 3:
                break

            # tolerance proportional to width of span; also enforce a floor
            tol = max(8, int((max(xs) - min(xs)) * 0.012))
            bins_now = _bin_xs(xs, tol)

            # need at least 3 bins to be "table-like"
            if len(bins_now) < 3:
                break

            if col_bins is None:
                col_bins = bins_now
                est_bins_history.append(bins_now)
                run = 1
                i += 1
                continue

            # Compare shape and alignment
            # allow off-by-1 in number of columns (due to merged/sparse cells)
            if abs(len(bins_now) - len(col_bins)) > 1:
                break

            # Align by position (left to right); compare average deviation
            m = min(len(bins_now), len(col_bins))
            diffs = [abs(col_bins[j] - bins_now[j]) for j in range(m)]
            mean_diff = sum(diffs) / m if m else 9999

            if mean_diff <= 22:  # a bit looser than before
                # update running estimate: use element-wise median across history + current
                est_bins_history.append(bins_now)
                # compute median length (favor the most common width)
                L = max(set(map(len, est_bins_history)), key=lambda L_: sum(1 for b in est_bins_history if len(b) == L_))
                # collect columns of chosen length and take column-wise medians
                chosen = [b for b in est_bins_history if len(b) == L]
                col_bins = [sorted(b[j] for b in chosen)[len(chosen)//2] for j in range(L)]
                run += 1
                i += 1
            else:
                break

        if run >= 3 and col_bins and len(col_bins) >= 3:
            tables.append((start, i - 1, col_bins))
        else:
            i = start + 1
    return tables


def rows_to_markdown(lines: List[List[Dict]]) -> str:
    paras = []
    buf = []
    last_y_mid = None
    last_h = 1
    scientific_units = re.compile(r"\b\d+(\.\d+)?\s*(°C|°F|%|mg/mL|µL|mL|g|kg|mm|cm|nm|μg|ng|mol/L)\b")

    for line in lines:
        if not line:
            continue
        text = " ".join(w["text"] for w in line).strip()
        if not text:
            continue
        y_mid = median((w["y1"] + w["y2"]) // 2 for w in line)
        h = max(1, line[0]["y2"] - line[0]["y1"])

        if scientific_units.search(text):
            # Force paragraph break before and after scientific lines
            if buf:
                paras.append(" ".join(buf))
                buf = []
            paras.append(text)
            last_y_mid = None
            continue

        if last_y_mid is not None and abs(y_mid - last_y_mid) > 2.0 * max(h, last_h):
            if buf:
                paras.append(" ".join(buf))
                buf = [text]
        else:
            buf.append(text)

        last_y_mid = y_mid
        last_h = h

    if buf:
        paras.append(" ".join(buf))
    return "\n\n".join(paras)


def table_lines_to_md(lines: List[List[Dict]], bins: Optional[List[float]] = None) -> str:
    if bins is None:
        all_centers = [(w["x1"] + w["x2"]) // 2 for line in lines for w in line]
        if not all_centers:
            return "> [Table detected, but could not parse]\n"
        span = max(all_centers) - min(all_centers)
        tol = max(8, int(span * 0.012))
        bins = _bin_xs(all_centers, tol)

    table_rows: List[List[str]] = []
    span = (bins[-1] - bins[0]) if len(bins) >= 2 else 1
    assign_tol = max(12, int(span * 0.02))

    for line in lines:
        row_cells: List[List[str]] = [[] for _ in bins]
        for w in line:
            cx = (w["x1"] + w["x2"]) // 2
            j = min(range(len(bins)), key=lambda k: abs(cx - bins[k]))
            row_cells[j].append(w["text"])
        row = [" ".join(cell).strip() for cell in row_cells]
        table_rows.append(row)

    def col_empty(j: int) -> bool:
        return all((not r[j].strip()) for r in table_rows)

    while table_rows and table_rows[0] and col_empty(0):
        for r in table_rows:
            del r[0]
        del bins[0]
    while table_rows and table_rows[0] and col_empty(len(table_rows[0]) - 1):
        for r in table_rows:
            r.pop()
        bins.pop()

    if not table_rows or not table_rows[0]:
        return "> [Table detected, but could not parse]\n"

    header = table_rows[0]
    letters = sum(ch.isalpha() for cell in header for ch in cell)
    digits = sum(ch.isdigit() for cell in header for ch in cell)
    has_header = letters >= digits and any(c.strip() for c in header)

    md = []
    if has_header:
        md.append("| " + " | ".join(c if c.strip() else f"Col{i+1}" for i, c in enumerate(header)) + " |")
        md.append("| " + " | ".join("---" for _ in header) + " |")
        rows_iter = table_rows[1:]
    else:
        cols = len(header)
        md.append("| " + " | ".join(f"Col{i+1}" for i in range(cols)) + " |")
        md.append("| " + " | ".join("---" for _ in range(cols)) + " |")
        rows_iter = table_rows

    for r in rows_iter:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)

def assemble_page_markdown(words: List[Dict]) -> str:
    lines = group_lines(words)
    table_spans = detect_table_blocks(lines)

    md_parts: List[str] = []
    used = [False] * len(lines)
    # mark table lines
    span_bins: Dict[Tuple[int, int], List[float]] = {}
    for (s, e, bins) in table_spans:
        for k in range(s, e + 1):
            used[k] = True
        span_bins[(s, e)] = bins

    i = 0
    while i < len(lines):
        is_table_block = used[i]
        j = i
        while j < len(lines) and used[j] == is_table_block:
            j += 1
        segment = lines[i:j]
        if is_table_block:
            # find which span this segment belongs to, to reuse its bins
            bins = None
            for (s, e), b in span_bins.items():
                if i >= s and j - 1 <= e:
                    bins = b
                    break
            md_parts.append(table_lines_to_md(segment, bins=bins))
        else:
            md_parts.append(rows_to_markdown(segment))
        md_parts.append("")  # blank line between blocks
        i = j

    return "\n".join(p for p in md_parts if p.strip())


# --- Main Loop ---
def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_path.glob("**/*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {INPUT_DIR}")
        sys.exit(1)

    print(f"📂 Found {len(pdf_files)} PDF(s) in {INPUT_DIR}")
    print(f"📝 Saving Markdown to {OUTPUT_DIR}")

    for pdf_file in pdf_files:
        process_pdf(pdf_file, output_path)

if __name__ == "__main__":
    main()
