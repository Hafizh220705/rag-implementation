# PharmaGuard-RAG

**Evidence-grounded Retrieval-Augmented Generation (RAG) untuk dokumen farmakovigilans BPOM berbahasa Indonesia.**

Prototipe riset ini dibuat untuk **KALBE Farma — AI Research Engineer Intern Technical Assessment, Option 1: RAG Implementation**. Sistem menerima pertanyaan pengguna, mencari bagian yang relevan dari dokumen BPOM, lalu menyajikan **kutipan evidence yang dipilih dari sumber** disertai identitas dokumen, halaman PDF, dan ID chunk. Seluruh retrieval dan inferensi model berjalan secara lokal; **Langfuse** dapat diaktifkan secara opsional untuk observability.

> **Status proyek:** pipeline RAG end-to-end, CLI, pengujian otomatis, evaluasi development set, dan live Langfuse tracing telah dijalankan. **Baseline evaluasi utama:** [`generation_baseline_v3_final.json`](outputs/evaluations/generation_baseline_v3_final.json). Prototipe ini bukan alat pengambilan keputusan klinis atau interpretasi hukum final.

## Daftar Isi

- [Fitur dan pendekatan](#fitur-dan-pendekatan)
- [Arsitektur sistem](#arsitektur-sistem)
- [Korpus dan provenance](#korpus-dan-provenance)
- [Technology stack](#technology-stack)
- [Quick start](#quick-start)
- [Cara menggunakan CLI](#cara-menggunakan-cli)
- [Evaluasi dan hasil](#evaluasi-dan-hasil)`
- [Langfuse observability](#langfuse-observability)
- [Struktur repository](#struktur-repository)
- [Keterbatasan dan pengembangan berikutnya](#keterbatasan-dan-pengembangan-berikutnya)
- [AI assistance disclosure](#ai-assistance-disclosure)

## Fitur dan pendekatan

- **Page-aware PDF ingestion:** ekstraksi PyMuPDF, audit halaman low-text, dan provenance per halaman.
- **Token-aware chunking:** 450 token per chunk dan overlap 75 token.
- **Semantic retrieval:** `intfloat/multilingual-e5-small` (384 dimensi, embedding ternormalisasi) + ChromaDB persistent collection dengan cosine distance.
- **Retrieval filtering:** `top_k=4`, minimum similarity `0.86`, serta pemenuhan dokumen yang disebutkan secara eksplisit dalam pertanyaan.
- **Evidence selection, bukan free-form synthesis (v3):** potongan evidence dibentuk dari teks halaman yang ditemukan; `qwen2.5:3b` melalui Ollama memilih ID evidence menggunakan keluaran JSON terstruktur. Jawaban akhir dirakit dari kutipan evidence terpilih, **bukan paragraf bebas yang dikarang model**.
- **Source-aware citations:** ID dokumen (`D1`–`D4`), halaman PDF, dan chunk ID dapat ditelusuri kembali ke korpus.
- **Controlled abstention:** pertanyaan tanpa bukti yang cukup diberi status `insufficient_evidence`; keluaran yang tidak memenuhi protokol divalidasi dan dapat dicoba ulang secara terbatas.
- **Evaluation artifacts:** baseline berversi, hasil retrieval, dan diagnosis kesalahan tanpa mengganti artefak eksperimen terdahulu.
- **Optional Langfuse tracing:** observasi retrieval dan generasi tercatat terpisah; kegagalan tracing tidak dirancang untuk menghentikan fungsi utama RAG.

**Catatan desain:** pembatasan jawaban pada kutipan sumber mengurangi risiko klaim bebas yang tidak didukung dokumen, tetapi **tidak menjamin evidence yang dipilih selalu paling relevan** dan belum melakukan sintesis analitis untuk semua pertanyaan perbandingan.

## Arsitektur sistem

```text
OFFLINE INDEXING
4 PDF BPOM + data/manifest.json
            |
            v
   PyMuPDF page extraction
            |
            +--> low-text page audit (tidak di-OCR)
            v
  Page-aware token chunking
  450 tokens, overlap 75
            |
            v
 multilingual-e5-small embeddings
            |
            v
 Persistent ChromaDB collection
   pharmaguard_dense_e5_v1

ONLINE QUESTION ANSWERING
       User question
            |
            v
 Dense retrieval (top_k=4)
 + explicit-document constraints
            |
            v
 Similarity / evidence gate ---------> insufficient_evidence
            |
            v
 Extract candidate evidence spans
            |
            v
 Ollama Qwen2.5:3b
 Select evidence IDs (structured JSON)
            |
            v
 Validate selection against candidates
            |
            v
 Render source excerpts + citations
 [document, PDF page, chunk ID]
            |
            v
 Answer + status + diagnostics

Optional Langfuse observations: dense-retrieval / ollama-generation
```

Implementasi berada di `src/pharmaguard_rag/`, termasuk `ingestion.py`, `chunking.py`, `embeddings.py`, `vector_store.py`, `retrieval_dense.py`, `evidence.py`, `prompting.py`, `generation.py`, `citations.py`, `pipeline.py`, `traced_pipeline.py`, dan `observability.py`.

## Korpus dan provenance

Korpus baseline berisi **4 dokumen publik BPOM, 296 halaman PDF, dan 348 chunk terindeks**. Sebanyak **18 halaman low-text** dicatat dalam audit tetapi tidak masuk indeks.

| ID | Dokumen | Peran dalam korpus | Sumber |
|---|---|---|---|
| **D1** | Peraturan BPOM Nomor 4 Tahun 2026 tentang Pelaksanaan Farmakovigilans | Rujukan regulasi; status berdasarkan metadata korpus per 20 Oktober 2026 | [JDIH BPOM](https://jdih.pom.go.id/view/slide/1738/4/2026/1d424cdb0df701a960adbb0092314569) |
| **D2** | Modul Farmakovigilans untuk Industri Farmasi (2020) | Referensi edukatif | [KMEI BPOM](https://kmei.pom.go.id/storage/medialain/Balai-Besar-POM-di-Jakarta-5eaa2719a581b_Modul%20Pelatihan%20Farmakovigilans%20untuk%20IF_oke.pdf) |
| **D3** | Modul Farmakovigilans Dasar (2020) | Referensi edukatif | [KMEI BPOM](https://kmei.pom.go.id/storage/medialain/KMEI-5eaa2698c3abc_Modul%20Farmakovigilans%20Dasar_.pdf) |
| **D4** | Buku Saku — Panduan Deteksi dan Pelaporan ESO untuk Tenaga Kesehatan (2019) | Referensi edukatif | [KMEI BPOM](https://kmei.pom.go.id/storage/medialain/Balai-Besar-POM-di-Jakarta-5c88d7a513d86_Buku%20Saku%20-%20Panduan%20Deteksi%20dan%20Pelaporan%20ESO%20untuk%20Tenaga%20Kesehatan.pdf) |

Nama file lokal, URL, penerbit, tahun, dan metadata lainnya disimpan dalam [`data/manifest.json`](data/manifest.json). Dokumen D2–D4 **bukan pengganti regulasi yang berlaku**. Untuk kewajiban regulatori, periksa juga sumber resmi terbarunya. Hak akses publik tidak otomatis berarti izin untuk mendistribusikan ulang PDF.

## Technology stack

| Komponen | Teknologi |
|---|---|
| Bahasa & runtime | Python 3.11 |
| PDF parsing | PyMuPDF |
| Chunking | Token-aware, page-aware |
| Embeddings | `intfloat/multilingual-e5-small` |
| Vector store | ChromaDB (persistent, cosine) |
| Generator / selector | Ollama + `qwen2.5:3b` |
| Testing | pytest |
| Observability | Langfuse (opsional) |
| Interface | Command-line interface (CLI) |

## Quick start

### Prasyarat

- Windows / PowerShell (lingkungan yang dipakai untuk baseline); Python **3.11** (`>=3.11,<3.13` dalam konfigurasi paket).
- Git, Ollama, dan ruang penyimpanan untuk embedding model, model LLM, PDF, serta indeks vektor.
- Akses internet diperlukan saat **mengunduh** paket/model/dokumen untuk pertama kali. Proses inference utama tidak memerlukan API LLM berbayar.

### 1. Instal dependensi dan model

Dari direktori root repository:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,models]"
Copy-Item .env.example .env
ollama pull qwen2.5:3b
python scripts/check_environment.py
```

Jalankan layanan Ollama jika belum aktif. Apabila aktivasi PowerShell diblokir, gunakan `\.venv\Scripts\python.exe` secara langsung. Model embedding akan diunduh dari Hugging Face saat pertama kali dipakai.

**Keamanan:** file `.env` mengandung konfigurasi lokal dan mungkin kredensial; jangan commit atau membagikannya. `.env.example` aman digunakan sebagai template tanpa nilai secret.

### 2. Unduh dan tempatkan PDF

Unduh **empat PDF** dari tautan dalam tabel korpus atau `data/manifest.json`. Simpan pada `data/raw/` menggunakan **nama file persis seperti pada manifest**. PDF mentah, hasil pemrosesan, dan database Chroma tidak disertakan dalam Git repository; buat ulang secara lokal.

### 3. Bangun korpus dan indeks

```powershell
python scripts/audit_corpus.py
python scripts/build_chunks.py
python scripts/validate_chunks.py
python scripts/build_index.py
```

`build_chunks.py` menghasilkan data halaman/chunk lokal dan laporan chunking. `build_index.py` membangun collection Chroma persisten dan melakukan pemeriksaan jumlah vektor. Jalankan tahap ini untuk **setup awal**; jangan menimpa artefak baseline historis hanya untuk pengujian ulang.

### Konfigurasi utama

Nilai runtime tersedia di `.env.example` dan/atau konfigurasi script.

| Variabel | Nilai baseline / contoh | Fungsi |
|---|---|---|
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Embedding model |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Endpoint Ollama lokal |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model pemilih evidence |
| `TOP_K` | `4` | Jumlah passage yang ditampilkan pada CLI |
| `LANGFUSE_ENABLED` | `false` | Tracing opsional |
| `LANGFUSE_PUBLIC_KEY` | (kosong) | Public key project |
| `LANGFUSE_SECRET_KEY` | (kosong) | Secret key project |
| `LANGFUSE_BASE_URL` | Sesuaikan host Langfuse project | Alamat layanan observability |

Konfigurasi eksperimen final: 450 token/chunk, overlap 75, embedding 384d ternormalisasi, cosine distance, `top_k=4`, relevance threshold `0.86`, collection `pharmaguard_dense_e5_v1`. Parameter generator dan hash dataset/korpus/kode direkam dalam artefak baseline final.

## Cara menggunakan CLI

**Pertanyaan faktual:**

```powershell
python scripts/ask.py "Apa yang dimaksud dengan pelaporan spontan?" --show-passages
```

**Pertanyaan lintas dokumen:**

```powershell
python scripts/ask.py "Bagaimana definisi farmakovigilans menurut Modul Dasar dan Modul untuk Industri Farmasi?" --show-passages
```

**Pertanyaan di luar cakupan:**

```powershell
python scripts/ask.py "Bagaimana cara mengganti oli mesin kendaraan?"
```

Untuk pertanyaan di luar korpus, perilaku yang diharapkan adalah `insufficient_evidence`, bukan menjawab dari pengetahuan umum LLM.

**Mode interaktif:**

```powershell
python scripts/ask.py
```

Ketik `quit` untuk keluar. Format jawaban pada v3 adalah kutipan evidence dengan sitasi seperti `[D3, PDF hlm. 31, D3_p031_c001]`. Nomor **halaman PDF** dapat berbeda dari nomor halaman yang tercetak di dalam dokumen.

## Evaluasi dan hasil

### Dataset dan protokol

Evaluasi development menggunakan **24 pertanyaan**: 8 faktual (`F01–F08`), 6 parafrasa (`P01–P06`), 6 lintas dokumen (`C01–C06`), dan 4 *out-of-domain* (`I01–I04`). Dataset ini dipakai selama pengembangan sistem; **bukan held-out test set**.

```powershell
python scripts/validate_evaluation_dataset.py
python scripts/run_retrieval_eval.py
```

**Hit@k, MRR@k, dan metrik retrieval berbasis gold tidak boleh dilaporkan sebagai skor valid** ketika gold evidence belum diverifikasi. Seluruh 24 kasus dalam baseline final masih mencatat `gold_verified=false`; retrieval report sebelumnya menampilkan denominator `0` dan nilai metrik `null` (bukan berarti skor 0%). Definisi metrik dan kebijakan anotasi ada di [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md).

### Perbandingan baseline development

| Indikator eksekusi | v1 | v2 | v3 | **v3_final** |
|---|---:|---:|---:|---:|
| Jumlah pertanyaan | 24 | 24 | 24 | **24** |
| `answered` | 8 | 19 | 19 | **20** |
| `insufficient_evidence` | 13 | 4 | 5 | **4** |
| `citation_error` | 1 | 1 | 0 | **0** |
| `execution_error` | 2 | 0 | 0 | **0** |
| Mean latency | 35,374 s | 7,642 s | 0,794 s | **0,352 s** |

**Interpretasi yang tepat:**

- **20/24 berstatus `answered`, bukan accuracy 83,3%.** Status hanya menyatakan pipeline menghasilkan output sesuai protokol, bukan bahwa jawaban benar.
- Seluruh **4 kasus out-of-domain** pada baseline final menghasilkan `insufficient_evidence`.
- Keenam kasus lintas dokumen (`C01–C06`) menyertakan sitasi dari **kedua dokumen yang diminta**, tetapi hal tersebut belum menjamin kualitas perbandingannya.
- v3 memperkenalkan **evidence selection/excerpt rendering**, sehingga perbandingan waktu dan jumlah output dengan v1/v2 tidak boleh dianggap perbandingan algoritme yang identik atau benchmark performa independen. Latensi juga bergantung pada keadaan model lokal, perangkat keras, dan lingkungan eksekusi.
- Audit jawaban **awal berbantuan AI**, berdasarkan evidence tersimpan, masih menemukan pemilihan kutipan yang kurang tepat. Ini **bukan validasi oleh anotator manusia independen** maupun pemeriksaan menyeluruh terhadap PDF asli.

Kasus yang perlu disebut secara terbuka:

- **F04:** gagal memilih pernyataan paling langsung bahwa **Pemilik Izin Edar** merupakan penanggung jawab utama (bukti relevan tersedia pada D1, halaman PDF 82).
- **P01:** evidence yang dipilih belum menjawab alasan utama perlunya pengawasan pasca-izin edar, yaitu keterbatasan informasi keamanan sebelum obat dipasarkan.
- **F07, P05, C02, C03, C05:** hasil masih dapat ditingkatkan dari sisi ketepatan, fokus, atau kelengkapan evidence (penilaian eksploratoris).

### Artefak evaluasi

- **Utama:** [`outputs/evaluations/generation_baseline_v3_final.json`](outputs/evaluations/generation_baseline_v3_final.json)
- Pembanding: [`generation_baseline_v3.json`](outputs/evaluations/generation_baseline_v3.json), [`generation_baseline_v2.json`](outputs/evaluations/generation_baseline_v2.json), dan [`generation_baseline_v1.json`](outputs/evaluations/generation_baseline_v1.json).
- Retrieval: [`retrieval_baseline_v2.json`](outputs/evaluations/retrieval_baseline_v2.json).
- Diagnosis historis: [`failure_analysis_v2.json`](outputs/evaluations/failure_analysis_v2.json) dan [`docs/failure_analysis_v2.md`](docs/failure_analysis_v2.md).

Untuk menjalankan evaluasi ulang, gunakan **nama output baru**; jangan menimpa baseline yang sudah digunakan dalam laporan:

```powershell
python scripts/run_generation_eval.py --all `
  --output outputs/evaluations/generation_recheck_local.json
```

Jalankan pengujian otomatis sebelum submission:

```powershell
$env:LANGFUSE_ENABLED="false"
python -m pytest -q -p no:cacheprovider
```

Pada pengujian lokal terakhir yang dilaporkan, **91 automated tests lulus**. Jalankan kembali di environment submission untuk memastikan hasil masih sama setelah perubahan lokal.

## Langfuse observability

Tracing opsional memungkinkan pemeriksaan proses RAG tanpa mengubah mekanisme jawaban. Integrasi live **telah teramati mengirim data ke project Langfuse**, termasuk root trace `pharmaguard-rag`, observation `dense-retrieval`, dan `ollama-generation`, serta input/output dan latensi.

Untuk mengaktifkannya, isi `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, dan `LANGFUSE_BASE_URL` di `.env` sesuai **host project milik Anda**, kemudian:

```powershell
python scripts/ask.py "Apa yang dimaksud dengan farmakovigilans?" --trace
```

Buka project Langfuse → **Tracing** → pilih trace `pharmaguard-rag` untuk memeriksa tahapan retrieval dan generation. Detail metadata yang dikirim bergantung pada implementasi observability dan konfigurasi runtime; jangan mengasumsikan setiap metrik atau token/cost otomatis tercatat.

**Privasi:** saat memakai Langfuse Cloud, data observasi mungkin dikirim keluar dari lingkungan lokal. Tinjau terlebih dahulu penggunaan data yang bersifat sensitif atau rahasia. Jangan mengunggah kredensial ke repository.

## Struktur repository

```text
pharmaguard-rag/
├── .env.example                 # Template konfigurasi non-secret
├── data/
│   ├── manifest.json           # Provenance 4 PDF
│   ├── raw/                    # PDF lokal (gitignored)
│   ├── processed/              # Hasil ingestion/chunking (gitignored)
│   └── indexes/                # ChromaDB persisten (gitignored)
├── src/pharmaguard_rag/        # Implementasi RAG modular
├── scripts/                    # CLI, indexing, evaluasi, audit, smoke tests
├── evaluation/datasets/        # Pertanyaan dan review eksperimen
├── outputs/
│   ├── evaluations/            # Baseline dan laporan JSON berversi
│   ├── examples/               # Sample output
│   └── figures/                # Visualisasi bila tersedia
├── tests/                      # Automated unit/integration-style tests
├── docs/                       # Metodologi, analisis kegagalan, catatan
├── pyproject.toml
└── README.md
```

Baseline eksperimen **dipertahankan secara terpisah** untuk menjaga riwayat pengembangan. File diagnostic dapat disimpan sebagai rekam jejak debugging dan bukan sumber utama angka performa.

## Keterbatasan dan pengembangan berikutnya

1. **Faithfulness ≠ relevance.** Kutipan yang benar berasal dari dokumen dapat tetap tidak menjawab inti pertanyaan; kasus F04 dan P01 menunjukkan hal ini.
2. **Belum ada gold evidence yang terverifikasi.** Tidak ada klaim akurasi, Hit@4, atau MRR@4 yang dapat dipertanggungjawabkan sebagai benchmark independen.
3. **Development-set tuning.** Threshold 0.86 dan strategi evidence dievaluasi pada dataset yang juga dipakai selama pengembangan; generalisasi ke pertanyaan baru belum dipastikan.
4. **Output ekstraktif.** Sistem sengaja memprioritaskan potongan teks sumber daripada jawaban sintetis yang lancar; hasil bisa panjang, kurang fokus, dan tidak otomatis menjelaskan hubungan lintas dokumen.
5. **Cakupan terbatas.** Hanya empat PDF; halaman image-only/low-text tidak di-OCR. Pembaruan peraturan dan materi lain memerlukan proses pembaruan korpus.
6. **Page provenance.** Halaman PDF dalam sitasi tidak selalu sama dengan nomor halaman tercetak.
7. **Domain safety.** Prototipe ini untuk eksplorasi dokumen, bukan dasar tunggal keputusan medis, keselamatan pasien, kepatuhan regulasi, maupun interpretasi hukum.

Arah pengembangan berikutnya: verifikasi gold evidence oleh reviewer independen, *held-out* evaluation set, peningkatan *evidence ranking/selection*, penyajian jawaban lebih ringkas berbasis bukti, penanganan dokumen image-only, dan pengujian robustness lintas parafrasa. Hal-hal tersebut adalah **future work**, bukan fitur yang diklaim sudah selesai.

## AI assistance disclosure

Asisten AI, digunakan untuk membantu drafting dan revisi kode, debugging, pembuatan pengujian, analisis hasil, dan dokumentasi. Seluruh keluaran AI memerlukan pemeriksaan; penilaian jawaban berbantuan AI **tidak diklaim sebagai human annotation independen**. Pemilik proyek bertanggung jawab atas kode, eksperimen, batasan, dan kebenaran informasi yang disampaikan saat submission.