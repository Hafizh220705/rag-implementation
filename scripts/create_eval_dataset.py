import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT / "evaluation" / "datasets"
    / "pharmaguard_eval_v1.json"
)

QUESTIONS = {
    "factual": [
        "Apa yang dimaksud dengan farmakovigilans?",
        "Mengapa keamanan obat yang beredar perlu terus dipantau?",
        "Apa yang dimaksud dengan pelaporan spontan?",
        "Siapa penanggung jawab utama aktivitas farmakovigilans menurut Peraturan BPOM Nomor 4 Tahun 2026?",
        "Apa kewajiban pemilik izin edar terkait penunjukan penanggung jawab farmakovigilans?",
        "Informasi apa yang harus disampaikan kepada Kepala Badan mengenai penanggung jawab farmakovigilans?",
        "Apa kontribusi laporan ESO terhadap deteksi sinyal keamanan obat?",
        "Bagaimana ketentuan kerja sama jika farmakovigilans melibatkan pihak lain?",
    ],
    "paraphrase": [
        "Mengapa pengawasan efek samping obat tidak berhenti setelah obat mendapat izin edar?",
        "Siapa yang harus menyiapkan personel khusus untuk mengurus pemantauan keamanan obat?",
        "Mengapa tenaga kesehatan perlu melaporkan dugaan efek samping obat?",
        "Apa yang perlu dilakukan perusahaan jika penanggung jawab farmakovigilans berubah?",
        "Mengapa industri farmasi memerlukan sistem untuk mengawasi risiko obat?",
        "Bagaimana farmakovigilans membantu mencegah masalah keamanan obat?",
    ],
    "cross_document": [
        "Bagaimana definisi farmakovigilans menurut Modul Dasar dan Modul untuk Industri Farmasi?",
        "Apa kesamaan tanggung jawab pemilik izin edar dalam Peraturan BPOM 4/2026 dan Modul Farmakovigilans Dasar?",
        "Bagaimana hubungan pelaporan spontan dalam Modul Dasar dengan manfaat laporan ESO dalam Buku Saku ESO?",
        "Apa tujuan pemantauan keamanan obat menurut Modul Industri Farmasi dan Modul Dasar?",
        "Bagaimana tanggung jawab industri farmasi dibahas dalam Modul Industri Farmasi dan Peraturan BPOM 4/2026?",
        "Bagaimana Modul Dasar dan Buku Saku ESO menjelaskan kontribusi pelaporan terhadap keamanan obat?",
    ],
    "insufficient_evidence": [
        "Bagaimana prosedur mengganti oli mesin kendaraan?",
        "Berapa harga saham Kalbe Farma pada tanggal 21 Oktober 2026?",
        "Apa jadwal jaga apoteker di Apotek Harmoni minggu ini?",
        "Berapa jumlah kasus ESO di Rumah Sakit ABC selama September 2026?",
    ],
}

PREFIXES = {
    "factual": "F",
    "paraphrase": "P",
    "cross_document": "C",
    "insufficient_evidence": "I",
}

def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Dataset already exists: {OUTPUT}. "
            "Refusing to overwrite annotations."
        )

    records = []

    for category, questions in QUESTIONS.items():
        for number, question in enumerate(
            questions,
            start=1,
        ):
            records.append({
                "question_id": (
                    f"{PREFIXES[category]}{number:02d}"
                ),
                "category": category,
                "question": question,
                "should_abstain": (
                    category == "insufficient_evidence"
                ),
                "gold_chunk_ids": [],
                "reference_answer": "",
                "verified": False,
                "review_notes": "",
            })

    assert len(records) == 24

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== EVALUATION DATASET CREATED ===")
    print("Total questions:", len(records))

    for category, questions in QUESTIONS.items():
        print(f"{category}: {len(questions)}")

    print("Verified questions: 0")
    print("Output:", OUTPUT)

if __name__ == "__main__":
    main()