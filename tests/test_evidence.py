import json

import pytest

from pharmaguard_rag.evidence import DOMAIN_TERMS, build_evidence, complete_units, render_selection


def page(text, doc="D1", title="Peraturan BPOM"):
    return {"text": text, "document_id": doc, "title": title,
            "chunk_id": f"{doc}_p001_c001", "pdf_page": 1}


@pytest.mark.parametrize("acronym, expansion", [
    ("KTD", "Kejadian Tidak Diinginkan"),
    ("ESO", "Efek Samping Obat"),
    ("KIPI", "Kejadian Ikutan Pasca Imunisasi"),
])
def test_terminology_is_literal_source_not_model_expansion(acronym, expansion):
    assert DOMAIN_TERMS[acronym] == expansion
    pages = [page(f"{expansion} ({acronym}) perlu dilaporkan kepada petugas.")]
    evidence = build_evidence(f"Apa {acronym}?", pages)
    answer, _, _ = render_selection('{"evidence_ids": ["E1"]}', evidence, pages, [])
    assert f"{expansion} ({acronym})" in answer


def test_adjacent_regulatory_obligations_keep_separate_scope():
    text = ("Tanggung jawab Pemilik Izin Edar adalah sebagai berikut: "
            "12. melaksanakan tindak lanjut regulatori yang ditetapkan Kepala Badan; "
            "13. memiliki perjanjian kerja sama dengan pihak lain;")
    units = complete_units(text)
    agreement = next(u for u in units if "perjanjian" in u["text"])
    assert "Pemilik Izin Edar" in agreement["text"]
    assert "Kepala Badan" not in agreement["text"]


@pytest.mark.parametrize("suffix", ["dan dapat", "dan dapat.", "kepada", "untuk", "termasuk"])
def test_dangling_clause_never_becomes_answer(suffix):
    assert complete_units("Penanggung jawab memahami profil keamanan " + suffix) == []


def test_trailing_fragment_and_leading_continuation_are_excluded():
    units = complete_units("s memahami profil keamanan obat. Nama dan kontak penanggung jawab wajib disampaikan. Pemilik Izin Edar harus")
    assert [u["text"] for u in units] == ["Nama dan kontak penanggung jawab wajib disampaikan."]


def test_offsets_reject_changed_source_even_for_valid_id():
    pages = [page("Pemilik Izin Edar wajib menyampaikan nama dan kontak.")]
    evidence = build_evidence("Nama kontak?", pages)
    pages[0]["text"] = "Tenaga kesehatan wajib menyampaikan nama dan kontak."
    with pytest.raises(ValueError, match="offsets"):
        render_selection('{"evidence_ids": ["E1"]}', evidence, pages, [])


def test_title_comes_from_metadata_not_generated_text():
    pages = [page("Farmakovigilans adalah kegiatan pemantauan keamanan obat.", "D2", "Modul Industri Farmasi"),
             page("Farmakovigilans adalah kegiatan pendeteksian masalah keamanan obat.", "D3", "Modul Dasar")]
    evidence = build_evidence("Definisi farmakovigilans?", pages)
    answer, sources, _ = render_selection('{"evidence_ids": {"D2": "E1", "D3": "E2"}}', evidence, pages, ["D2", "D3"])
    assert "Modul Industri Farmasi:" in answer.splitlines()[0]
    assert "Modul Dasar:" in answer.splitlines()[1]
    assert [s["document_id"] for s in sources] == ["D2", "D3"]


def test_missing_required_source_in_selection_fails_closed():
    pages = [page("Farmakovigilans adalah kegiatan pemantauan keamanan obat.", "D2"),
             page("Farmakovigilans adalah kegiatan pemantauan keamanan obat.", "D3")]
    evidence = build_evidence("Farmakovigilans?", pages)
    with pytest.raises(ValueError):
        render_selection('{"evidence_ids": {"D2": "E1"}}', evidence, pages, ["D2", "D3"])
    answer, _, selected = render_selection('{"evidence_ids": {"D2": "E1", "D3": null}}', evidence, pages, ["D2", "D3"])
    assert answer == "" and selected == []


def test_overlength_selection_is_rejected_not_cut_mid_sentence():
    pages = [page("Pemilik Izin Edar " + "informasi " * 75 + "wajib dilaporkan."),
             page("Tenaga kesehatan " + "informasi " * 75 + "wajib dilaporkan.", "D3")]
    evidence = build_evidence("Informasi?", pages)
    with pytest.raises(ValueError, match="120 words"):
        render_selection('{"evidence_ids": ["E1", "E2"]}', evidence, pages, [])


def test_query_morphology_keeps_change_deadline_as_candidate():
    pages = [page("Setiap perubahan penanggung jawab harus dilaporkan paling lambat satu bulan. "
                  "Penanggung jawab mengelola sistem farmakovigilans sesuai ketentuan.")]
    evidence = build_evidence("Jika penanggung jawab berubah?", pages)
    assert "satu bulan" in evidence[0]["text"]


def test_numbered_heading_is_not_a_complete_proposition():
    text = ("B. Definisi dan Ruang Lingkup Farmakovigilans "
            "1. Definisi dan Tujuan dari PV 2. Farmakovigilans adalah kegiatan "
            "pemantauan keamanan penggunaan obat.")
    units = complete_units(text)
    assert len(units) == 1
    assert "adalah kegiatan" in units[0]["text"]
    assert "Tujuan dari PV" not in units[0]["text"]


def test_definition_does_not_offer_a_purpose_or_audit_as_definition():
    pages = [page("Tujuan farmakovigilans adalah mendeteksi masalah keamanan obat. "
                  "Farmakovigilans adalah kegiatan pemantauan keamanan obat. "
                  "Audit farmakovigilans merupakan bagian penting dari sistem mutu.")]
    evidence = build_evidence("Bagaimana definisi farmakovigilans menurut D1?", pages)
    assert len(evidence) == 1
    assert evidence[0]["text"].startswith("Farmakovigilans adalah")


def test_designation_question_does_not_offer_unrelated_personnel_duties():
    pages = [page("Pemilik Izin Edar harus menunjuk penanggung jawab dan wakil. "
                  "Pemilik Izin Edar harus memberikan pelatihan kepada penanggung jawab.")]
    evidence = build_evidence("Apa kewajiban terkait penunjukan penanggung jawab?", pages)
    assert len(evidence) == 1
    assert "menunjuk" in evidence[0]["text"]


def test_span_ranking_considers_all_retrieved_pages_before_packing():
    pages = [page("Farmakovigilans merupakan bagian dari sistem keamanan obat."),
             page("Pemilik Izin Edar harus memiliki perjanjian kerja sama dengan pihak lain.")]
    evidence = build_evidence("Apa ketentuan perjanjian kerja sama pihak lain dalam farmakovigilans?", pages)
    assert "perjanjian kerja sama" in evidence[0]["text"]


def test_named_recipient_and_requested_action_must_be_present():
    pages = [page("Industri farmasi harus menyampaikan nama dan kontak kepada Badan POM. "
                  "Pemilik Izin Edar harus menyampaikan nama dan kontak kepada Kepala Badan. "
                  "Perubahan personil harus dilaporkan kepada Kepala Badan.")]
    evidence = build_evidence("Informasi apa yang harus disampaikan kepada Kepala Badan?", pages)
    assert len(evidence) == 1
    assert "Pemilik Izin Edar harus menyampaikan" in evidence[0]["text"]


def test_side_effect_list_does_not_explain_ongoing_monitoring():
    pages = [page("Efek samping yang umum terjadi berupa mual dan sakit kepala.")]
    assert build_evidence("Mengapa pengawasan efek samping harus terus dilakukan?", pages) == []
    pages.append(page("Keterbatasan data keamanan menyebabkan tidak semua risiko obat dapat diketahui."))
    evidence = build_evidence("Mengapa keamanan obat harus terus dipantau?", pages)
    assert len(evidence) == 1
    assert "Keterbatasan" in evidence[0]["text"]
