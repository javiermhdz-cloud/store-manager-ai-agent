import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).resolve().parent.parent / "eval" / "run_eval.py"
)
run_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_eval)
grade = run_eval.grade


def test_numbers_pass_with_commas_and_currency():
    question = {"numbers": [[927240.99, 1]]}
    status, reasons, _ = grade(question, "La venta neta fue de $927,241 MXN en julio.")
    assert status == "PASS" and not reasons


def test_missing_number_fails():
    question = {"numbers": [[837, 0], [27, 0]]}
    status, reasons, _ = grade(question, "Vendió $837.00 MXN.")
    assert status == "FAIL"
    assert any("27" in reason for reason in reasons)


def test_refusal_expected_and_given_passes():
    question = {"expect_refusal": True}
    status, _, _ = grade(question, "No tengo esa información. Los datos terminan el 31 de agosto.")
    assert status == "PASS"


def test_refusal_expected_but_answer_given_fails():
    question = {"expect_refusal": True}
    status, _, _ = grade(question, "Se venderán 500,000 pesos.")
    assert status == "FAIL"


def test_false_refusal_is_flagged():
    question = {"numbers": [[3, 0]]}
    status, reasons, _ = grade(question, "No tengo esa información.")
    assert status == "FAIL"
    assert "rechazo indebido" in reasons


def test_service_error_is_not_counted_as_wrong_answer():
    question = {"numbers": [[3, 0]]}
    for message in (
        "El servicio del modelo está saturado, intenta de nuevo en un momento.",
        "No pude consultar la información disponible. Verifica la configuración e inténtalo de nuevo.",
    ):
        status, _, _ = grade(question, message)
        assert status == "SERVICE_ERROR"


def test_citation_needs_document_and_section_mark():
    question = {"numbers": [[48, 0]], "cite_docs_any": ["devoluciones", "faq"]}
    ok, _, cited_ok = grade(question, "48 horas (Procedimiento de devoluciones y cambios v4.0, §3).")
    assert ok == "PASS" and cited_ok is True
    bad, reasons, cited_bad = grade(question, "48 horas según el procedimiento de devoluciones.")
    assert bad == "FAIL" and cited_bad is False
    assert any("cita" in reason for reason in reasons)


def test_contains_groups_are_and_of_ors_and_ignore_accents():
    question = {"contains": [["7:00", "7 am"], ["22:00", "10 pm"]]}
    status, _, _ = grade(question, "Atiende de 7:00 a 22:00 horas.")
    assert status == "PASS"
    status, reasons, _ = grade(question, "Abre a las 7:00.")
    assert status == "FAIL" and reasons
    question = {"contains": [["tortillas de maiz"]]}
    status, _, _ = grade(question, "Tortillas de maíz 1 kg")
    assert status == "PASS"


def test_questions_file_is_well_formed():
    import json

    questions = json.loads(
        (Path(__file__).resolve().parent.parent / "eval" / "questions.json").read_text(encoding="utf-8")
    )
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    for q in questions:
        assert q["question"].strip()
        assert q.get("expect_refusal") or q.get("numbers") or q.get("contains")


def test_number_words_count_as_numbers():
    status, _, _ = grade({"numbers": [[3, 0]]}, "Hay tres productos por debajo del punto de reorden.")
    assert status == "PASS"
    status, _, _ = grade({"numbers": [[90, 0]]}, "La compra no debe tener más de noventa días.")
    assert status == "PASS"
    status, _, _ = grade({"numbers": [[17, 0]]}, "Son diecisiete lotes.")
    assert status == "PASS"
    assert 35.0 in run_eval.numbers_in("un descuento de treinta y cinco por ciento")


def test_wrong_number_word_still_fails():
    status, _, _ = grade({"numbers": [[3, 0]]}, "Hay cuatro productos.")
    assert status == "FAIL"


def test_number_words_are_accepted_for_small_integers():
    question = {"numbers": [[90, 0]]}
    status, _, _ = grade(question, "Procede si la compra tiene menos de noventa días.")
    assert status == "PASS"
    question = {"numbers": [[3, 0]]}
    status, _, _ = grade(question, "Hay tres productos por debajo del punto de reorden.")
    assert status == "PASS"
    question = {"numbers": [[17, 0]]}
    status, _, _ = grade(question, "Son diecisiete productos.")
    assert status == "PASS"
    question = {"numbers": [[72, 0]]}
    status, _, _ = grade(question, "Son setenta y dos tickets.")
    assert status == "PASS"


def test_number_words_do_not_accept_the_wrong_number():
    question = {"numbers": [[3, 0]]}
    status, _, _ = grade(question, "Hay cuatro productos.")
    assert status == "FAIL"
    question = {"numbers": [[927240.99, 1]]}
    status, _, _ = grade(question, "La venta fue de novecientos mil pesos.")
    assert status == "FAIL"