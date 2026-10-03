from datetime import date

import pandas as pd
import pytest

import src.data_tools as data_tools
from src.config import INVENTORY_SNAPSHOT_DATE, SALES_END_DATE, SALES_START_DATE
from src.data_tools import (
    list_tickets,
    lots_expiring_soon,
    rank_skus,
    sales_summary,
    search_tickets,
    sku_info,
    stock_below_reorder,
    ticket_counts,
)


def test_config_contains_data_date_bounds():
    # The expected dates are the coverage dates documented in the data dictionary.
    assert SALES_START_DATE == date(2026, 6, 1)
    assert SALES_END_DATE == date(2026, 8, 31)
    assert INVENTORY_SNAPSHOT_DATE == date(2026, 9, 1)


@pytest.mark.parametrize(
    ("group_by", "start_date", "end_date", "expected"),
    [
        # ventas.csv row for 2026-06-01, T01, SKU-1001: 52 units, MXN 1,430.
        ("day", "2026-06-01", "2026-06-01", [{"dia": "2026-06-01", "unidades": 52, "venta_neta_mxn": 1430.0}]),
        # Sum T01/SKU-1001 rows June 1-7: 259 units and MXN 6,938.30.
        ("week", "2026-06-01", "2026-06-07", [{"semana_inicio": "2026-06-01", "unidades": 259, "venta_neta_mxn": 6938.3}]),
        # Sum T01/SKU-1001 rows in June and July; it has no August sales.
        ("month", "2026-06-01", "2026-08-31", [
            {"mes": "2026-06", "unidades": 1197, "venta_neta_mxn": 32139.95},
            {"mes": "2026-07", "unidades": 576, "venta_neta_mxn": 15552.29},
            {"mes": "2026-08", "unidades": 0, "venta_neta_mxn": 0.0},
        ]),
    ],
)
def test_sales_summary_groups_by_calendar_period(group_by, start_date, end_date, expected):
    result = sales_summary(
        start_date, end_date, store_id="T01", sku="SKU-1001", group_by=group_by
    )
    assert result["results"] == expected
    assert result["filters"]["include_returns"] is True


def test_sales_summary_groups_by_store_category_and_channel():
    # For June 1, store totals are direct sums of ventas.csv rows for each tienda_id.
    by_store = sales_summary("2026-06-01", "2026-06-01", group_by="store")["results"]
    assert {row["tienda_id"]: row["unidades"] for row in by_store} == {
        "T01": 1497, "T02": 1458, "T03": 625, "T04": 508
    }

    # SKU-1002's June 1 T01 channel rows are 7 + 31 units and MXN 203 + 705.48.
    by_channel = sales_summary(
        "2026-06-01", "2026-06-01", store_id="T01", sku="SKU-1002", group_by="channel"
    )["results"]
    assert {row["canal"]: row["unidades"] for row in by_channel} == {
        "Ecommerce": 7, "Piso": 31
    }
    assert {row["canal"]: row["venta_neta_mxn"] for row in by_channel} == {
        "Ecommerce": 203.0, "Piso": 705.48
    }

    # Those same SKU-1002 rows total 38 units and MXN 908.48 in category Lácteos.
    by_category = sales_summary(
        "2026-06-01", "2026-06-01", store_id="T01", sku="SKU-1002", group_by="category"
    )["results"]
    assert by_category == [
        {"categoria": "Lácteos", "unidades": 38, "venta_neta_mxn": 908.48}
    ]


def test_sales_summary_return_default_and_exclusion():
    # T01/SKU-1041 on June 1 has three rows: Ecommerce 2/62, Piso 26/806, Piso -1/-31 (a return).
    included = sales_summary(
        "2026-06-01", "2026-06-01", store_id="T01", sku="SKU-1041"
    )["results"]
    excluded = sales_summary(
        "2026-06-01", "2026-06-01", store_id="T01", sku="SKU-1041", include_returns=False
    )["results"]
    assert included[0]["unidades"] == 27
    assert included[0]["venta_neta_mxn"] == 837.0
    assert excluded[0]["unidades"] == 28
    assert excluded[0]["venta_neta_mxn"] == 868.0


def test_sales_summary_rejects_dates_outside_csv_coverage():
    # The documented sales CSV starts June 1; May 31 must not produce a fabricated result.
    with pytest.raises(ValueError, match="Sales dates"):
        sales_summary("2026-05-31", "2026-06-01")


def test_rank_skus_uses_csv_units_and_catalog_fields():
    # On June 1, T01 dairy SKU-1001 has 52 units; the next highest (SKU-1011) has 46.
    result = rank_skus(
        "2026-06-01", "2026-06-01", metric="units", store_id="T01", category="Lácteos", limit=1
    )
    assert result["results"][0]["sku"] == "SKU-1001"
    assert result["results"][0]["unidades"] == 52
    assert result["results"][0]["nombre"] == "Leche entera 1 L"


def test_sku_info_includes_discontinued_substitute_and_stock():
    # The catalog lists SKU-1001's substitute as SKU-1121; no inventory row exists for it.
    result = sku_info("SKU-1001")
    match = result["matches"][0]
    assert match["catalog"]["estatus"] == "Descontinuado"
    assert match["catalog"]["sku_sustituto"] == "SKU-1121"
    assert match["substitute"]["sku"] == "SKU-1121"
    assert match["stock_by_store"] == []
    assert result["inventory_snapshot_date"] == "2026-09-01"


def test_stock_below_reorder_uses_snapshot_and_deficit():
    # inventario.csv has T01/SKU-1013 at 72 units vs. a reorder point of 260: deficit 188.
    result = stock_below_reorder(store_id="T01")
    row = next(item for item in result["results"] if item["sku"] == "SKU-1013")
    assert row["existencia"] == 72
    assert row["punto_reorden"] == 260
    assert row["faltante_reorden"] == 188
    assert result["total"] == 3
    assert result["filters"]["inventory_snapshot_date"] == "2026-09-01"


def test_stock_below_reorder_includes_row_exactly_at_reorder_point(monkeypatch):
    # inventario.csv has no row with existencia == punto_reorden, so this uses a crafted fixture.
    fixture = pd.DataFrame(
        [
            {
                "tienda_id": "T01",
                "sku": "SKU-9001",
                "existencia": 100,
                "punto_reorden": 100,
                "fecha_ultimo_conteo": pd.Timestamp("2026-08-26"),
                "fecha_caducidad_lote_proximo": pd.NaT,
            }
        ]
    )
    monkeypatch.setattr(data_tools, "_load_inventory", lambda: fixture)
    result = stock_below_reorder(store_id="T01")
    row = next(item for item in result["results"] if item["sku"] == "SKU-9001")
    assert row["existencia"] == 100
    assert row["punto_reorden"] == 100
    assert row["faltante_reorden"] == 0
    assert result["total"] == 1


def test_lots_expiring_soon_counts_from_snapshot_date():
    # T01/SKU-1015 expires Sep 5 and SKU-1003 Sep 7, both within six days of Sep 1.
    result = lots_expiring_soon(within_days=6, store_id="T01")
    expiring = {row["sku"] for row in result["results"]}
    assert {"SKU-1015", "SKU-1003"} <= expiring
    assert result["total"] == 10
    assert all(row["fecha_caducidad_lote_proximo"] <= "2026-09-07" for row in result["results"])


def test_lots_expiring_soon_flags_already_expired_lots(monkeypatch):
    # inventario.csv has no lot earlier than the Sep 1 snapshot, so this uses a crafted fixture.
    fixture = pd.DataFrame(
        [
            {
                "tienda_id": "T01",
                "sku": "SKU-9002",
                "existencia": 10,
                "punto_reorden": 5,
                "fecha_ultimo_conteo": pd.Timestamp("2026-08-26"),
                "fecha_caducidad_lote_proximo": pd.Timestamp("2026-08-20"),
            }
        ]
    )
    monkeypatch.setattr(data_tools, "_load_inventory", lambda: fixture)
    result = lots_expiring_soon(within_days=6, store_id="T01")
    row = next(item for item in result["results"] if item["sku"] == "SKU-9002")
    assert row["ya_caducado"] is True
    assert result["total"] == 1


@pytest.mark.parametrize(
    ("group_by", "store_id", "result_key", "expected"),
    [
        # tickets_mesa_servicio.csv rows for T01 grouped by categoria.
        ("category", "T01", "categoria", {"Cajas": 11, "Inventario": 14, "Mantenimiento": 12, "RH": 12, "TI": 23}),
        # Same T01 rows grouped by prioridad.
        ("priority", "T01", "prioridad", {"Alta": 13, "Baja": 26, "Crítica": 1, "Media": 32}),
        # Same T01 rows grouped by estado.
        ("state", "T01", "estado", {"Abierto": 2, "Cerrado": 53, "En proceso": 3, "Resuelto": 14}),
        # All tickets (no store filter) grouped by tienda_id.
        ("store_id", None, "tienda_id", {"T01": 72, "T02": 72, "T03": 90, "T04": 51}),
    ],
)
def test_ticket_counts_by_group(group_by, store_id, result_key, expected):
    result = ticket_counts(group_by=group_by, store_id=store_id)
    counts = {row[result_key]: row["count"] for row in result["results"]}
    assert counts == expected
    assert result["total"] == sum(expected.values())


def test_list_tickets_filters_structured_fields():
    # Ticket 1001 is a T01 Inventario/Media/Resuelto ticket created June 2 at 10:15.
    result = list_tickets(
        start_date="2026-06-02",
        end_date="2026-06-02",
        store_id="T01",
        category="Inventario",
        priority="Media",
        state="Resuelto",
    )
    assert [row["ticket_id"] for row in result["results"]] == [1001]
    assert result["total"] == 1


def test_list_tickets_total_precedes_limit():
    result = list_tickets(store_id="T01", limit=1)

    assert result["total"] == 72
    assert len(result["results"]) == 1


def test_search_tickets_fuzzy_matches_description():
    # Ticket 1001's description contains "platano y fresa"; accents are normalized for matching.
    result = search_tickets("plátano y fresa", store_id="T01", category="Inventario")
    assert result["results"][0]["ticket_id"] == 1001


def test_search_tickets_any_mode_finds_leak_tickets():
    # "de" is dropped as a stopword; match_mode="any" finds tickets mentioning "fuga" or "agua".
    result = search_tickets("fuga de agua", match_mode="any", limit=100)
    assert result["total_matches"] == 12
    for row in result["results"]:
        combined = f"{row['titulo']} {row['descripcion']}".lower()
        assert "fuga" in combined or "agua" in combined


def test_search_tickets_finds_caja_typo_variants():
    # 90 of 285 tickets contain "caja" or its plural "cajas" in titulo or descripcion, including ticket 1033.
    result = search_tickets("caja", limit=100)
    assert result["total_matches"] == 90
    assert any(row["ticket_id"] == 1033 for row in result["results"])


def test_search_tickets_plural_matches_same_tickets_as_singular():
    # Stripping the trailing "s" makes "cajas" normalize to the same token as "caja".
    singular = search_tickets("caja", limit=100)
    plural = search_tickets("cajas", limit=100)
    assert plural["total_matches"] == singular["total_matches"]
    assert {row["ticket_id"] for row in plural["results"]} == {
        row["ticket_id"] for row in singular["results"]
    }


def test_search_tickets_nonsense_query_has_no_matches():
    # No ticket text can fuzzy- or exact-match a word that does not resemble real Spanish.
    result = search_tickets("xyzxyzxyz")
    assert result["total_matches"] == 0
    assert result["results"] == []


def test_search_tickets_rejects_text_of_only_stopwords():
    with pytest.raises(ValueError, match="stopwords"):
        search_tickets("de la y en")