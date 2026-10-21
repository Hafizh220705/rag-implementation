
from pathlib import Path
import fitz

DOCUMENTS = {
    "D1": ("D1_perbpom_4_2026.pdf", [71]),
    "D2": ("D2_modul_industri_farmasi_2020.pdf", [7, 13]),
    "D3": (
        "D3_modul_dasar_2020.pdf",
        [1, 4, 5, 7, 9, 26, 27, 28, 41, 42],
    ),
    "D4": (
        "D4_buku_saku_eso_2019.pdf",
        [20, 21, 22, 24, 25],
    ),
}

output_dir = Path("outputs/audit_review")
output_dir.mkdir(parents=True, exist_ok=True)

for doc_id, (filename, pages) in DOCUMENTS.items():
    path = Path("data/raw") / filename

    with fitz.open(path) as pdf:
        for page_number in pages:
            page = pdf[page_number - 1]
            text = page.get_text("text")

            image_path = output_dir / (
                f"{doc_id}_page_{page_number}.png"
            )
            text_path = output_dir / (
                f"{doc_id}_page_{page_number}.txt"
            )

            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(1.5, 1.5),
                alpha=False
            )

            pixmap.save(str(image_path))
            text_path.write_text(
                text,
                encoding="utf-8"
            )

            print(
                f"{doc_id} | Page {page_number} | "
                f"{len(text)} characters"
            )

print("\nInspection files generated.")
