import pytest

from src.policies import search_policies


def test_search_policies_finds_defect_exception_section():
    # procedimiento_devoluciones.pdf p.3, section 4.1, states a non-perishable
    # manufacturing defect is accepted even past the normal return window,
    # within 90 days and with Customer Service Manager authorization.
    results = search_policies(
        "¿Qué pasa si un producto tiene defecto de fabricación después del plazo de devolución?"
    )
    assert any(
        row["document"] == "procedimiento_devoluciones.pdf"
        and row["section"] == "4.1 Producto con defecto de fabricación"
        for row in results
    )


def test_search_policies_finds_dairy_shrinkage_tolerance_section():
    # politica_mermas_y_caducidad.pdf p.3, section 4, has the tolerance table
    # listing Lácteos at 1.8% monthly tolerance with a 2.8% alert threshold.
    results = search_policies(
        "¿Cuál es la tolerancia mensual de merma para la categoría lácteos?"
    )
    assert results[0]["document"] == "politica_mermas_y_caducidad.pdf"
    assert results[0]["section"] == "4 Tabla de tolerancias de merma por categoría"
    assert results[0]["page"] == 3


def test_search_policies_finds_power_outage_backup_section():
    # manual_apertura_y_cierre_tienda.pdf p.5, section 7.1, says the backup
    # generator starts automatically within a maximum of thirty seconds.
    results = search_policies("¿Cuánto tiempo tarda en arrancar la planta de emergencia?")
    assert results[0]["document"] == "manual_apertura_y_cierre_tienda.pdf"
    assert results[0]["section"] == "7.1 Respaldo disponible"
    assert results[0]["page"] == 5


def test_search_policies_finds_card_terminal_ticket_priority_section():
    # manual_apertura_y_cierre_tienda.pdf p.7, section 9.2, lists "Terminal
    # bancaria rechaza todas las tarjetas" with priority "Alta".
    results = search_policies(
        "¿Qué prioridad tiene el ticket si la terminal bancaria rechaza todas las tarjetas?"
    )
    assert results[0]["document"] == "manual_apertura_y_cierre_tienda.pdf"
    assert results[0]["section"] == "9.2 Otras incidencias frecuentes"
    assert results[0]["page"] == 7


def test_search_policies_returns_empty_list_for_unrelated_query():
    # None of the four policy PDFs mention these words, so no chunk scores above zero.
    assert search_policies("xylophone intergalactic quasar blorf") == []


def test_search_policies_dairy_removal_ranks_formal_policy_or_faq_first():
    # politica_mermas_y_caducidad.pdf p.4, section 5.2, states dairy products
    # are removed from the shelf two days before their expiration date; the
    # FAQ's "3 Mermas y perecederos" section covers the same topic more loosely.
    results = search_policies(
        "¿Cuántos días antes de la fecha de caducidad se debe retirar la leche del anaquel?"
    )
    top = results[0]
    assert (
        top["document"] == "politica_mermas_y_caducidad.pdf"
        and top["section"] == "5.2 Regla específica para lácteos"
    ) or top["document"] == "faq_gerentes_de_tienda.pdf"
    assert all(not row["section"].endswith("Control de cambios") for row in results)
    assert all(row["version"] for row in results)


def test_search_policies_excludes_control_de_cambios_sections():
    # "Control de cambios" is a changelog table, not answerable policy content.
    assert all(
        not row["section"].endswith("Control de cambios")
        for row in search_policies("¿Qué cambios ha tenido este documento?")
    )


def test_search_policies_rejects_text_of_only_stopwords():
    with pytest.raises(ValueError, match="stopwords"):
        search_policies("de la y en")
