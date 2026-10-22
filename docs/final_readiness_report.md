# PharmaGuard-RAG — Final Improvement, Evaluation, and Submission Readiness

**Pembaruan terakhir:** 22 Oktober 2026  
**Assessment:** KALBE AI Research Engineer Intern — Option 1: RAG Implementation  
**Baseline utama:** `outputs/evaluations/generation_baseline_v3_final.json`  
**Status:** **Conditionally ready for technical submission and demo**; belum layak mengklaim akurasi terverifikasi.

> **Status dokumen:** Ini adalah laporan kesiapan terbaru untuk baseline **v3_final**. Hasil v1, v2, dan v3 awal dipertahankan sebagai riwayat eksperimen, bukan disamakan dengan implementasi aktif. README adalah panduan instalasi dan penggunaan; dokumen ini merangkum bukti, temuan, serta batasan kesiapan.

## 1. Ringkasan eksekutif

PharmaGuard-RAG adalah sistem tanya jawab berbasis *Retrieval-Augmented Generation* lokal untuk empat dokumen publik farmakovigilans BPOM. Alur ingest–index–retrieve–select evidence–render citations–abstain dapat dijalankan melalui CLI. Integrasi tracing Langfuse juga telah diuji secara langsung.

Baseline `v3_final` mengevaluasi **24 pertanyaan development set**: **20 `answered`**, **4 `insufficient_evidence`**, **0 `citation_error`**, dan **0 `execution_error`**. Empat penolakan merupakan kasus out-of-domain I01–I04. Enam pertanyaan cross-document C01–C06 menyertakan sitasi dari **kedua dokumen yang diminta**.

Hasil tersebut adalah **status operasional dan cakupan sumber**, **bukan** *answer accuracy*, *faithfulness score*, ataupun metrik retrieval terhadap gold evidence. Audit awal berbantuan AI masih menemukan jawaban tidak tepat, terutama **F04** dan **P01**. Seluruh `gold_verified` tetap `false`; belum ada verifikasi independen terhadap PDF sumber untuk mengesahkan gold evidence. Dataset ini juga dipakai selama pengembangan, bukan *held-out benchmark*.

## 2. Lingkup, corpus, dan konfigurasi yang dipertahankan

| Komponen | Baseline |
|---|---|
| Corpus | 4 PDF publik BPOM; 296 halaman PDF; 18 halaman low-text tidak diindeks; 348 chunk |
| Ingestion | PyMuPDF, ekstraksi sadar halaman, audit low-text |
| Chunking | 450 token, overlap 75 token |
| Embedding | `intfloat/multilingual-e5-small` (384 dimensi, normalized) |
| Vector store | ChromaDB persisten, cosine distance, collection `pharmaguard_dense_e5_v1` |
| Retrieval | Dense retrieval, `top_k=4`, minimum similarity `0.86`, batas dokumen eksplisit bila diminta |
| LLM lokal | Ollama `qwen2.5:3b` |
| Mode jawaban v3 | `evidence_excerpts`: model memilih ID evidence terstruktur; aplikasi menampilkan kutipan dan sitasi sumber |
| Observability | Langfuse opsional, tracing tidak wajib untuk menjalankan RAG |
| Data evaluasi | 8 factual (F), 6 paraphrase (P), 6 cross-document (C), 4 insufficient/out-of-domain (I) |

Dokumen sumber: **D1** Peraturan BPOM Nomor 4 Tahun 2026; **D2** Modul Farmakovigilans untuk Industri Farmasi (2020); **D3** Modul Farmakovigilans Dasar (2020); **D4** Buku Saku Panduan Deteksi dan Pelaporan ESO untuk Tenaga Kesehatan (2019). URL, nama file, dan provenance ada di `data/manifest.json`. Modul lama adalah materi edukatif, bukan pengganti peraturan terkini.

### Identitas eksperimen yang terekam

- Corpus SHA-256: `cb936746ed6314f85aca570217927374c859012b6b0beb470ddeee03580652db`.
- Dataset SHA-256: `df070333bca35623be96f2607b20b8345b9115f4e2a8e1369a192a7eca01090c`.
- Collection: `pharmaguard_dense_e5_v1`.
- Konfigurasi generasi yang direkam untuk `v3_final`: `num_ctx=8192`, `num_predict=128`, `temperature=0`, `seed=42`, `repeat_penalty=1.1`, maksimal **2 generation attempts**.

Hash tersebut diambil dari metadata artefak `v3_final`, bukan bukti bahwa seluruh kode atau indeks lokal saat ini byte-identical dengan saat eksperimen. Baseline dan indeks lama jangan ditimpa saat mencoba ulang.

## 3. Perkembangan arsitektur dan pengamanan

### Tahap v1 → v2 (historis)

1. Pengetatan **abstention** dan relevance gate 0.86, dengan penanganan token abstention yang tidak valid.
2. Validasi sitasi dan keluaran **fail-closed**; output salah format tidak diberikan sebagai jawaban final.
3. Penanganan truncation (`done_reason=length`) dan retry terbatas pada generasi.
4. Pelaporan evaluasi dengan hash dataset/corpus, latensi, penyebab kegagalan, dan validasi gold evidence yang ketat.
5. CLI, instruksi setup, provenance corpus, serta evaluasi retrieval dan failure analysis.

Hasil v2 penting sebagai **riwayat**: C05 masih mengalami `citation_error`; beberapa pertanyaan cross-document dan atribusi dokumen belum andal. Temuan ini tidak boleh ditulis sebagai kegagalan eksekusi aktif pada `v3_final`.

### Tahap v3 → v3_final (implementasi aktif)

1. **Evidence extraction** menghasilkan kandidat span yang terikat pada passage hasil retrieval.
2. Qwen2.5 memilih **ID evidence** melalui output terstruktur, bukan menyusun jawaban regulasi bebas dalam bahasa natural.
3. Renderer menyajikan **kutipan evidence** lengkap dengan judul dokumen, halaman PDF, dan chunk ID; validasi menolak pilihan evidence yang tidak sah.
4. Retrieval mempertimbangkan **dokumen yang secara eksplisit diminta**, terutama untuk pertanyaan cross-document.
5. Bila evidence tidak memadai, sistem **abstain**. Desain ini mengurangi peluang klaim bebas yang tidak bersumber, tetapi tidak menjamin bahwa evidence yang dipilih selalu relevan.

**Trade-off:** respons dapat berupa beberapa kutipan panjang yang tidak menyimpulkan relasi antar-sumber; *faithful excerpt* tidak otomatis berarti menjawab pertanyaan secara tepat.

## 4. Hasil evaluasi development set

| Hasil pipeline | v1 | v2 | v3 | **v3_final** |
|---|---:|---:|---:|---:|
| Jumlah pertanyaan | 24 | 24 | 24 | **24** |
| `answered` | 8 | 19 | 19 | **20** |
| `insufficient_evidence` | 13 | 4 | 5 | **4** |
| `citation_error` | 1 | 1 | 0 | **0** |
| `execution_error` | 2 | 0 | 0 | **0** |
| Mean latency | 35,374 s | 7,642 s | 0,794 s | **0,352 s** |

**Cara membaca tabel:**

- `answered` menandakan pipeline menghasilkan output sesuai protokol. **20/24 tidak sama dengan akurasi 83,3%**.
- I01–I04: seluruhnya `insufficient_evidence`, sesuai harapan pada empat pertanyaan di luar cakupan.
- C01–C06: keenam jawaban final menyertakan **kedua dokumen yang diminta**. Ini bukti *document inclusion*, **bukan** recall terhadap gold passage atau ketepatan perbandingan.
- Latensi dicatat pada eksekusi lokal dan dipengaruhi warm cache, panjang output, serta perubahan protokol generasi. Perbandingan v1/v2 dengan v3 **bukan benchmark throughput terkontrol**.
- Hit@k, MRR@k, dan metrik kualitas berbasis gold **belum dapat diklaim**, karena seluruh 24 kasus `gold_verified=false`.

### Review awal kualitas jawaban (berbantuan AI)

| Kategori review | Jumlah | Kasus |
|---|---:|---|
| Tepat | 13 | F01, F02, F03, F05, F06, F08, P02, P03, P04, P06, C01, C04, C06 |
| Sebagian tepat | 5 | F07, P05, C02, C03, C05 |
| Tidak tepat | 2 | **F04, P01** |
| Abstain sesuai | 4 | I01, I02, I03, I04 |

Penilaian ini merupakan **screening berbantuan AI terhadap kutipan dan evidence tersimpan**, bukan hasil anotator manusia independen. Karena belum ada verifikasi menyeluruh ke PDF asli, kategori di atas tidak dipublikasikan sebagai *human-verified correctness/faithfulness metric*.

### Temuan prioritas

- **F04 — salah memilih fokus evidence.** Jawaban membahas akses/kualifikasi personel penanggung jawab, tetapi tidak menyebut langsung pihak **Pemilik Izin Edar** sebagai penanggung jawab utama. Passage yang lebih langsung ada di D1, PDF hlm. 82 (`D1_p082_c001`).
- **P01 — alasan pemantauan pascapemasaran tidak terjawab langsung.** Kutipan dari D1 hlm. 128 membahas komunikasi informasi keamanan baru, bukan keterbatasan data keamanan pra-pemasaran. Bukti yang lebih relevan terlihat pada D1 hlm. 125 (`D1_p125_c001`) dan D2 hlm. 2 (`D2_p002_c001`).
- **F07 — evidence tambahan kurang fokus.** D1 hlm. 124 relevan untuk sinyal, tetapi D4 hlm. 19 yang dipilih berisi petunjuk pelaporan; D4 hlm. 13 lebih langsung membahas deteksi dini sinyal.
- **P05, C02, C03, C05 — sebagian tepat.** Evidence berasal dari sumber yang berkaitan, tetapi alasan atau hubungan lintas dokumen belum selalu diekspresikan secara lengkap. Untuk C05, tidak ada lagi `citation_error` seperti pada v2, namun perbandingannya masih sempit.
- **C06 — pulih dari false abstention pada v3.** `v3_final` menyajikan D3 hlm. 31 dan D4 hlm. 13. Ini memperbaiki cakupan sumber pada kasus tersebut, tetapi tetap perlu evaluasi manusia untuk menilai kecukupan jawaban.

Detail keterbatasan aktif dan prioritasnya dijabarkan di [`docs/known_issues.md`](known_issues.md).

## 5. Bukti verifikasi teknis dan observability

### Automated tests

Pengujian terakhir di lingkungan Windows/Python 3.11.9 yang dilaporkan:

```text
91 passed in 2.49s
```

Perintah reproduksi:

```powershell
$env:LANGFUSE_ENABLED="false"
python -m pytest -q -p no:cacheprovider
```

Suite ini memverifikasi komponen seperti ingestion, chunking, retrieval, parsing sitasi, evidence, pipeline, evaluation validation, dan wrapper tracing. **Unit tests yang lulus tidak membuktikan seluruh jawaban farmakovigilans benar**.

### Environment dan CLI

`python scripts/check_environment.py` berhasil memeriksa Python 3.11.9 serta dependensi utama yang terpasang; pemeriksaan dependensi saja tidak membuktikan koneksi Ollama. Demo end-to-end sebelumnya sudah dijalankan dan diamati pada Langfuse.

Contoh perintah:

```powershell
python scripts/ask.py "Apa yang dimaksud dengan farmakovigilans?" --show-passages
python scripts/ask.py "Bagaimana prosedur mengganti oli mobil?"
```

### Langfuse

Pada pemeriksaan langsung 22 Oktober 2026, project Langfuse milik pengguna menerima root trace `pharmaguard-rag`, dengan observasi terpisah **`dense-retrieval`** dan **`ollama-generation`**, serta input/output dan waktu eksekusi. Integrasi dapat dinonaktifkan untuk menjalankan pipeline secara lokal.

Contoh pengujian tracing:

```powershell
python scripts/ask.py "Apa itu farmakovigilans?" --trace
```

Tracing cloud memerlukan kredensial di `.env` yang tidak dilacak Git. Rekaman trace membuktikan instrumentasi berjalan; **tidak membuktikan factual correctness**, dan tidak berarti semua metadata tambahan atau pengukuran biaya model pasti tercatat.

### Pemeriksaan Git (status saat audit)

- `.gitignore` sudah mengabaikan `.env`, `.venv`, data mentah, data hasil pemrosesan, dan indeks Chroma.
- `git check-ignore -v .env` menunjukkan `.env` diabaikan; `git ls-files .env` kosong.
- `git diff --check` tidak menunjukkan kesalahan whitespace, tetapi menampilkan peringatan konversi LF/CRLF pada Windows; peringatan itu bukan kegagalan pengujian.
- Banyak perubahan masih belum di-*stage*, termasuk kode v3 dan artefak evaluasi. **Status repo belum otomatis final hanya karena tes lulus**; perlu pemeriksaan file dan secret sebelum pengguna memutuskan commit/push.

## 6. Artefak dan provenance hasil

| Jenis | File |
|---|---|
| **Baseline utama saat ini** | `outputs/evaluations/generation_baseline_v3_final.json` |
| Pembanding v3 awal | `outputs/evaluations/generation_baseline_v3.json` |
| Riwayat v1/v2 | `outputs/evaluations/generation_baseline_v1.json`; `generation_baseline_v2.json` |
| Retrieval evaluation | `outputs/evaluations/retrieval_baseline_v2.json` |
| Failure analysis v1/v2 | `outputs/evaluations/failure_analysis_v1.json`; `failure_analysis_v2.json` |
| Protokol | `docs/evaluation_protocol.md` |
| Review awal berbantuan AI | Artefak review terpisah; **bukan** perubahan pada `gold_verified` dalam dataset |
| Riwayat analisis v2 | `docs/baseline_failure_analysis.md`; `docs/failure_analysis_v2.md` |

Tiga file `generation_diagnostic_v3_*.json` adalah artefak eksplorasi. Simpan untuk penelusuran jika diperlukan, tetapi jangan menampilkan angka dari file diagnostik sebagai hasil utama. Jangan menghapus/menimpa baseline terdahulu untuk menyelaraskan angka.

## 7. Checklist kesiapan submission

| Area | Status | Bukti/batasan |
|---|---|---|
| PDF ingestion dan chunking | Siap | 296 halaman; 348 chunk; 18 halaman low-text tidak diindeks |
| Embedding dan persistent index | Siap | E5-small dan ChromaDB |
| Retrieval dan document constraints | Siap untuk demo | Semua C01–C06 menyertakan kedua dokumen, kualitas evidence tidak selalu optimal |
| Answer rendering dan citations | Siap untuk demo | 20 answered, 0 citation_error pada development run v3_final |
| Abstention | Siap untuk contoh OOD | I01–I04 abstain; belum diuji pada held-out set |
| Automated tests | Lulus pada pengujian terakhir | 91/91 |
| Langfuse observability | Teruji langsung | Root trace + tahap retrieval/generation terlihat |
| README dan metode evaluasi | Tersedia | README v3_final dan protokol evaluasi |
| Human-verified correctness & gold evidence | **Belum selesai** | Seluruh `gold_verified=false` |
| Held-out evaluation | **Belum dilakukan** | Belum ada estimasi generalisasi independen |
| GitHub artifact review / submission | **Perlu finalisasi** | File `??`, sinkronisasi docs, pemeriksaan secret dan staging |

### Tindakan berikutnya sebelum submission

1. Periksa apakah README, `known_issues.md`, dan laporan ini mengacu ke **baseline v3_final**; v1/v2 ditempatkan jelas sebagai riwayat.
2. Audit daftar file baru yang perlu diikutkan (`scripts/ask.py`, `src/pharmaguard_rag/evidence.py`, tes, output baseline) dan file diagnostik yang opsional; periksa kredensial sebelum staging.
3. Verifikasi reproducibility dari petunjuk README, siapkan demo faktual, cross-document, out-of-domain, serta contoh trace Langfuse.
4. Untuk klaim kualitas di masa mendatang, lakukan validasi gold evidence terhadap PDF asli dan uji held-out baru. **Jangan membuat angka accuracy, Hit@4, atau MRR@4 dari status `answered`.**

## 8. Keterbatasan dan keputusan akhir

Sistem hanya mencakup empat dokumen dan teks yang berhasil diekstrak; halaman image-only belum di-OCR. Penanda `pdf_page` merujuk halaman file PDF, bukan selalu angka yang tercetak. Evaluasi memakai pertanyaan yang sama selama pengembangan, threshold similarity 0.86 belum divalidasi terhadap distribusi baru, dan keluaran ekstraktif dapat kurang menjawab pertanyaan walaupun kutipannya benar.

**Keputusan:** PharmaGuard-RAG **layak dipresentasikan sebagai prototipe RAG yang berjalan, terobservasi, dan memiliki jejak eksperimen**. Klaim kualitas jawaban yang sudah terverifikasi secara independen **belum layak dibuat**. Penggunaan untuk keputusan klinis, keselamatan pasien, atau kepatuhan hukum/regulasi memerlukan pemeriksaan langsung ke sumber resmi dan penilaian profesional.

## 9. AI assistance disclosure

AI Assistance digunakan untuk membantu drafting/perbaikan kode, debugging, analisis evaluasi, serta dokumentasi. Review AI tidak disamakan dengan anotasi manusia. Pemilik proyek tetap bertanggung jawab atas kode, keakuratan eksperimen, dokumentasi, dan materi yang diserahkan.
