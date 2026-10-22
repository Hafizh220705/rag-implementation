# Known Issues — PharmaGuard-RAG (Baseline v3_final)

**Diperbarui:** 22 Oktober 2026  
**Baseline utama:** `outputs/evaluations/generation_baseline_v3_final.json`  
**Tujuan:** Mencatat keterbatasan **aktif** pada implementasi v3_final. Temuan khusus v1/v2 dicatat terpisah sebagai riwayat; jangan menganggapnya kegagalan terkini.

## Status singkat

- Development set: **24 kasus** (`F01–F08`, `P01–P06`, `C01–C06`, `I01–I04`).
- Status v3_final: **20 `answered`**, **4 `insufficient_evidence`**, **0 `citation_error`**, **0 `execution_error`**.
- **6/6** kasus cross-document menyertakan kedua dokumen yang diminta. Ini hanya cakupan sumber, **bukan** nilai kebenaran jawaban.
- Review awal **berbantuan AI**: 13 tepat, 5 sebagian tepat, 2 tidak tepat, 4 abstain sesuai. Bukan *human-verified accuracy*.
- Seluruh kasus `gold_verified=false`; evaluasi menggunakan **development set**, bukan *held-out benchmark*.

## Masalah aktif

### KI-001 — Faithfulness dan gold evidence belum diverifikasi independen

**Prioritas: tinggi untuk klaim kualitas; tidak menghalangi demo.**

- Sitasi yang valid secara struktural hanya membuktikan format/referensi sumber dapat dilacak, bukan kecukupan bukti untuk menjawab pertanyaan.
- Review awal merupakan screening berbantuan AI terhadap evidence yang tersimpan, bukan verifikasi menyeluruh ke halaman PDF asli.
- **Dampak:** skor factual correctness, faithfulness, Hit@k, dan MRR@k belum boleh diklaim sebagai hasil terverifikasi.
- **Tindak lanjut:** reviewer independen memeriksa PDF sumber, mengisi gold chunk dan reference answer, baru mengubah status verified sesuai protokol pada `docs/evaluation_protocol.md`.

### KI-002 — F04: evidence tidak menyebut penanggung jawab utama secara langsung

**Prioritas: tinggi — kualitas jawaban.**

- **Pertanyaan:** Siapa penanggung jawab utama aktivitas farmakovigilans menurut Peraturan BPOM Nomor 4 Tahun 2026?
- **Output v3_final:** memilih D1 PDF hlm. 26 tentang akses dan kualifikasi penanggung jawab/personel farmakovigilans.
- **Masalah:** tidak mengidentifikasi secara langsung **Pemilik Izin Edar** sebagai penanggung jawab utama.
- **Bukti kandidat yang lebih tepat:** D1 PDF hlm. 82, `D1_p082_c001` (tersedia di retrieval untuk kasus lain).
- **Tindak lanjut:** audit *evidence ranking/selection*; jangan menulis jawaban yang diperbaiki manual ke file baseline lama.

### KI-003 — P01: alasan pengawasan pasca-izin edar kurang terjawab

**Prioritas: tinggi — kualitas jawaban.**

- **Pertanyaan:** Mengapa pengawasan efek samping obat tidak berhenti setelah obat mendapat izin edar?
- **Output v3_final:** kutipan D1 hlm. 128 tentang komunikasi informasi keamanan baru.
- **Masalah:** belum menyatakan langsung bahwa data keamanan pada uji klinik/pra-pemasaran terbatas sehingga risiko tertentu baru teridentifikasi setelah obat digunakan lebih luas.
- **Bukti kandidat yang lebih tepat:** D1 PDF hlm. 125 (`D1_p125_c001`) atau D2 PDF hlm. 2 (`D2_p002_c001`).
- **Tindak lanjut:** uji *relevance* pemilihan kutipan pada pertanyaan alasan/sebab dan variasi parafrasa.

### KI-004 — Kutipan valid tetapi sebagian jawaban kurang fokus atau lengkap

**Prioritas: sedang — kualitas dan keterbacaan.**

- **F07:** evidence D1 hlm. 124 relevan tentang deteksi sinyal, tetapi evidence D4 hlm. 19 yang ikut dipilih lebih membahas prosedur pelaporan; D4 hlm. 13 lebih langsung mengenai manfaat pelaporan ESO.
- **P05:** dua kutipan D2 hlm. 21 menyentuh kepatuhan dan manajemen risiko, tetapi alasan kebutuhan sistem dapat disampaikan lebih jelas.
- **C02, C03, C05:** kedua dokumen terwakili, namun pilihan kutipan dan penjelasan hubungan/perbandingan masih parsial; C05 cenderung memilih kewajiban operasional yang sempit.
- **Akar trade-off:** v3 menggunakan *evidence ID selection* dan penyajian kutipan, **bukan sintesis jawaban bebas**. Sistem dapat memilih teks yang benar tetapi kurang menjawab intent.
- **Tindak lanjut:** riset reranking evidence, *query-intent alignment*, dan ringkasan yang tetap terikat pada sumber; perlakukan sebagai eksperimen baru, bukan perubahan baseline diam-diam.

### KI-005 — Generalisasi dan metrik retrieval belum teruji pada held-out set

**Prioritas: sedang–tinggi untuk klaim performa.**

- Threshold minimum similarity **0.86** disetel/diuji pada 24 pertanyaan yang digunakan selama development.
- I01–I04 berhasil abstain di development set, tetapi belum membuktikan robust terhadap domain lain atau parafrasa baru.
- Karena gold chunk belum verified, nilai Hit@4/MRR@4 **tidak dapat diestimasi secara sah** dari dataset ini.
- **Tindak lanjut:** susun held-out questions yang tidak dipakai untuk tuning; lakukan verifikasi gold dan evaluasi dengan denominator yang jelas.

### KI-006 — Cakupan corpus, OCR, dan konteks regulatori terbatas

**Prioritas: sedang.**

- Corpus hanya mencakup 4 PDF dan 348 chunk; 18 halaman low-text tidak dimasukkan ke indeks. Halaman image-only belum di-OCR.
- Nomor `pdf_page` adalah halaman PDF, bukan selalu halaman tercetak.
- D2/D3 (2020) dan D4 (2019) merupakan bahan edukasi, bukan otoritas yang menggantikan Peraturan BPOM Nomor 4 Tahun 2026 (D1).
- **Tindak lanjut:** pertimbangkan OCR, pembaruan korpus/regulasi, serta validasi sumber sebelum digunakan untuk keputusan nyata.

### KI-007 — Angka latency dan observability memiliki batas interpretasi

**Prioritas: sedang untuk pelaporan eksperimen.**

- Mean latency v3_final **0,352 s** berasal dari satu development run lokal; perubahan generasi dari v1/v2 ke v3, kondisi cache, dan status Ollama membuat perbandingan antarrun tidak setara.
- Trace Langfuse langsung berhasil merekam root `pharmaguard-rag`, `dense-retrieval`, dan `ollama-generation`, tetapi **trace bukan bukti akurasi**.
- Mengaktifkan Langfuse Cloud dapat mengirimkan prompt/passage ke layanan eksternal; hindari data rahasia tanpa review privasi.
- **Tindak lanjut:** dokumentasikan kondisi benchmark bila pengukuran performa terkontrol dibutuhkan.

## Riwayat masalah v1/v2 — bukan status aktif v3_final

| Masalah historis | Status pada v3_final |
|---|---|
| C02 pada v2 tidak mencakup kedua dokumen | **Document inclusion teratasi**: C02 memakai D1 dan D3; kualitas perbandingan masih parsial (KI-004) |
| C05 pada v2 menghasilkan `citation_error`/marker `[BsN]` | **Tidak muncul** pada v3_final; C05 `answered` dengan D1 dan D2, tetapi masih parsial (KI-004) |
| C01 pada v2 berisiko salah mengatribusikan judul modul | C01 v3_final menyertakan atribusi D2 dan D3 yang tepat menurut metadata terindeks; verifikasi sumber independen tetap tertunda (KI-001) |
| F08 pada v2 mengandung detail yang diduga unsupported | F08 v3_final sudah menggunakan kutipan perjanjian kerja sama dari D1 dan D3; review independen tetap diperlukan |
| C06 pada v3 awal `insufficient_evidence` (false abstention) | C06 v3_final `answered` dengan D3 dan D4; kasus ini tidak lagi false abstention pada run final |

## Batas penggunaan dan status keputusan

PharmaGuard-RAG **siap untuk demonstrasi teknis** sebagai prototipe riset yang dapat mengutip sumber dan menolak pertanyaan di luar corpus. Proyek **belum** boleh mengklaim akurasi klinis atau benchmark kebenaran jawaban yang telah diverifikasi. Jawaban tidak boleh digunakan sebagai satu-satunya dasar keputusan medis atau interpretasi kepatuhan regulatori. Lihat `README.md`, `docs/final_readiness_report.md`, dan artefak `generation_baseline_v3_final.json` untuk konteks lengkap.
