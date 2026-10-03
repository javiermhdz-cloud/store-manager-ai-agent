"""Deterministic queries over the store's CSV data."""

from __future__ import annotations

import difflib
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Literal

import pandas as pd

from src.config import (
    DATA_DIR,
    INVENTORY_SNAPSHOT_DATE,
    SALES_END_DATE,
    SALES_START_DATE,
)

SalesGroup = Literal["day", "week", "month", "store", "category", "channel"]
SalesMetric = Literal["sales_mxn", "units"]
TicketGroup = Literal["category", "priority", "state", "store_id"]

SPANISH_STOPWORDS = {
    "de", "la", "el", "en", "los", "las", "un", "una", "y", "a",
    "que", "se", "no", "con", "por", "para",
}


def _as_date(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _validate_sales_range(start_date: date | str, end_date: date | str) -> tuple[date, date]:
    start = _as_date(start_date)
    end = _as_date(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    if start < SALES_START_DATE or end > SALES_END_DATE:
        raise ValueError(
            f"Sales dates must be between {SALES_START_DATE} and {SALES_END_DATE}"
        )
    return start, end


def _validate_ticket_range(
    start_date: date | str | None, end_date: date | str | None
) -> tuple[date | None, date | None]:
    start = _as_date(start_date) if start_date is not None else None
    end = _as_date(end_date) if end_date is not None else None
    if start and end and start > end:
        raise ValueError("start_date must be on or before end_date")
    if (start and start < SALES_START_DATE) or (end and end > SALES_END_DATE):
        raise ValueError(
            f"Ticket dates must be between {SALES_START_DATE} and {SALES_END_DATE}"
        )
    return start, end


def _load_sales() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ventas.csv", parse_dates=["fecha"])


def _load_products() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "catalogo_productos.csv")


def _load_inventory() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "inventario.csv", parse_dates=["fecha_ultimo_conteo", "fecha_caducidad_lote_proximo"])


def _load_tickets() -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / "tickets_mesa_servicio.csv",
        parse_dates=["fecha_creacion", "fecha_cierre"],
    )


def _clean_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        # Date-only columns parse as midnight timestamps; drop the time in that case.
        if value == value.normalize():
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _clean_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {key: _clean_value(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _filter_sales(
    start_date: date | str,
    end_date: date | str,
    store_id: str | None,
    category: str | None,
    sku: str | None,
    channel: str | None,
    include_returns: bool,
) -> tuple[pd.DataFrame, date, date]:
    start, end = _validate_sales_range(start_date, end_date)
    sales = _load_sales()
    sales = sales[sales["fecha"].between(pd.Timestamp(start), pd.Timestamp(end))]
    if store_id is not None:
        sales = sales[sales["tienda_id"] == store_id]
    if sku is not None:
        sales = sales[sales["sku"] == sku]
    if channel is not None:
        sales = sales[sales["canal"] == channel]
    if not include_returns:
        sales = sales[sales["unidades"] >= 0]
    if category is not None:
        products = _load_products()[["sku", "categoria"]]
        sales = sales.merge(products, on="sku", how="left")
        sales = sales[sales["categoria"] == category]
    return sales, start, end


def _period_labels(start: date, end: date, group_by: SalesGroup) -> list[str]:
    if group_by == "day":
        dates = pd.date_range(start=start, end=end, freq="D")
        return [value.date().isoformat() for value in dates]
    if group_by == "week":
        first_monday = start - timedelta(days=start.weekday())
        last_monday = end - timedelta(days=end.weekday())
        dates = pd.date_range(start=first_monday, end=last_monday, freq="7D")
        return [value.date().isoformat() for value in dates]
    first_month = start.replace(day=1)
    last_month = end.replace(day=1)
    dates = pd.date_range(start=first_month, end=last_month, freq="MS")
    return [value.strftime("%Y-%m") for value in dates]


def sales_summary(
    start_date: date | str,
    end_date: date | str,
    store_id: str | None = None,
    category: str | None = None,
    sku: str | None = None,
    channel: str | None = None,
    include_returns: bool = True,
    group_by: SalesGroup = "day",
) -> dict[str, object]:
    """Aggregate net sales and units by a time period or sales dimension."""
    valid_groups = {"day", "week", "month", "store", "category", "channel"}
    if group_by not in valid_groups:
        raise ValueError(f"group_by must be one of {sorted(valid_groups)}")
    sales, start, end = _filter_sales(
        start_date, end_date, store_id, category, sku, channel, include_returns
    )

    if group_by in {"day", "week", "month"}:
        group_column = {"day": "dia", "week": "semana_inicio", "month": "mes"}[group_by]
        sales[group_column] = sales["fecha"].map(
            lambda value: (
                (value.date() - timedelta(days=value.weekday())).isoformat()
                if group_by == "week"
                else value.strftime("%Y-%m")
                if group_by == "month"
                else value.date().isoformat()
            )
        )
    else:
        group_column = {
            "store": "tienda_id",
            "category": "categoria",
            "channel": "canal",
        }[group_by]
        if group_by == "category" and "categoria" not in sales.columns:
            sales = sales.merge(_load_products()[["sku", "categoria"]], on="sku", how="left")

    grouped = (
        sales.groupby(group_column, dropna=False, as_index=False)
        .agg(unidades=("unidades", "sum"), venta_neta_mxn=("venta_neta_mxn", "sum"))
        .sort_values(group_column)
    )
    if group_by in {"day", "week", "month"}:
        grouped = grouped.set_index(group_column).reindex(
            _period_labels(start, end, group_by), fill_value=0
        )
        grouped.index.name = group_column
        grouped = grouped.reset_index()
    grouped["unidades"] = grouped["unidades"].map(lambda value: int(value) if value.is_integer() else value)
    return {
        "filters": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "store_id": store_id,
            "category": category,
            "sku": sku,
            "channel": channel,
            "include_returns": include_returns,
            "group_by": group_by,
        },
        "results": _clean_records(grouped),
    }


def rank_skus(
    start_date: date | str,
    end_date: date | str,
    metric: SalesMetric = "sales_mxn",
    direction: Literal["top", "bottom"] = "top",
    limit: int = 10,
    store_id: str | None = None,
    category: str | None = None,
    include_returns: bool = True,
) -> dict[str, object]:
    """Rank products by net sales or units during the requested period."""
    if metric not in {"sales_mxn", "units"}:
        raise ValueError("metric must be 'sales_mxn' or 'units'")
    if direction not in {"top", "bottom"}:
        raise ValueError("direction must be 'top' or 'bottom'")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    sales, start, end = _filter_sales(
        start_date, end_date, store_id, category, None, None, include_returns
    )
    grouped = sales.groupby("sku", as_index=False).agg(
        unidades=("unidades", "sum"), venta_neta_mxn=("venta_neta_mxn", "sum")
    )
    products = _load_products()[["sku", "nombre", "categoria"]]
    grouped = grouped.merge(products, on="sku", how="left")
    value_column = "venta_neta_mxn" if metric == "sales_mxn" else "unidades"
    grouped = grouped.sort_values(
        [value_column, "sku"], ascending=[direction == "bottom", True]
    ).head(limit)
    return {
        "filters": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "metric": metric,
            "direction": direction,
            "limit": limit,
            "store_id": store_id,
            "category": category,
            "include_returns": include_returns,
        },
        "results": _clean_records(grouped),
    }


def sku_info(sku_or_name: str) -> dict[str, object]:
    """Find catalog products by SKU or name, with snapshot stock and substitute."""
    query = sku_or_name.strip()
    if not query:
        raise ValueError("sku_or_name cannot be empty")
    products = _load_products()
    normalized_query = _normalize_text(query)
    exact_sku = products[products["sku"].str.casefold() == query.casefold()]
    exact_name = products[products["nombre"].map(_normalize_text) == normalized_query]
    if not exact_sku.empty:
        matches = exact_sku
    elif not exact_name.empty:
        matches = exact_name
    else:
        contains = products[
            products["nombre"].map(lambda value: normalized_query in _normalize_text(value))
        ]
        if not contains.empty:
            matches = contains
        else:
            scored = products.assign(
                _score=products["nombre"].map(
                    lambda value: difflib.SequenceMatcher(
                        None, normalized_query, _normalize_text(value)
                    ).ratio()
                )
            )
            matches = scored[scored["_score"] >= 0.45].sort_values(
                ["_score", "sku"], ascending=[False, True]
            ).head(10)

    inventory = _load_inventory()
    result_matches = []
    for product in matches.to_dict(orient="records"):
        stock = inventory[inventory["sku"] == product["sku"]][
            ["tienda_id", "existencia", "punto_reorden", "fecha_ultimo_conteo"]
        ]
        substitute_sku = product.get("sku_sustituto")
        substitute_row = products[products["sku"] == substitute_sku]
        result_matches.append(
            {
                "catalog": {key: _clean_value(value) for key, value in product.items() if key != "_score"},
                "stock_by_store": _clean_records(stock),
                "substitute": (
                    {key: _clean_value(value) for key, value in substitute_row.iloc[0].items()}
                    if not substitute_row.empty
                    else None
                ),
            }
        )
    return {"query": query, "inventory_snapshot_date": INVENTORY_SNAPSHOT_DATE.isoformat(), "matches": result_matches}


def stock_below_reorder(
    store_id: str | None = None, category: str | None = None
) -> dict[str, object]:
    """List snapshot inventory at or below its reorder point."""
    inventory = _load_inventory()
    products = _load_products()[["sku", "nombre", "categoria"]]
    stock = inventory.merge(products, on="sku", how="left")
    stock = stock[stock["existencia"] <= stock["punto_reorden"]]
    if store_id is not None:
        stock = stock[stock["tienda_id"] == store_id]
    if category is not None:
        stock = stock[stock["categoria"] == category]
    total = len(stock)
    stock = stock.assign(
        faltante_reorden=stock["punto_reorden"] - stock["existencia"],
        fecha_corte=INVENTORY_SNAPSHOT_DATE.isoformat(),
    ).sort_values(["tienda_id", "sku"])
    return {
        "filters": {
            "store_id": store_id,
            "category": category,
            "inventory_snapshot_date": INVENTORY_SNAPSHOT_DATE.isoformat(),
        },
        "total": total,
        "results": _clean_records(stock),
    }


def lots_expiring_soon(
    within_days: int = 30,
    store_id: str | None = None,
    category: str | None = None,
) -> dict[str, object]:
    """List perishable lots expiring within N days of the inventory snapshot, including already-expired lots."""
    if within_days < 0:
        raise ValueError("within_days cannot be negative")
    cutoff = INVENTORY_SNAPSHOT_DATE + timedelta(days=within_days)
    inventory = _load_inventory()
    inventory = inventory[
        inventory["fecha_caducidad_lote_proximo"] <= pd.Timestamp(cutoff)
    ]
    inventory = inventory.assign(
        ya_caducado=inventory["fecha_caducidad_lote_proximo"] < pd.Timestamp(INVENTORY_SNAPSHOT_DATE)
    )
    products = _load_products()[["sku", "nombre", "categoria", "perecedero"]]
    lots = inventory.merge(products, on="sku", how="left")
    if store_id is not None:
        lots = lots[lots["tienda_id"] == store_id]
    if category is not None:
        lots = lots[lots["categoria"] == category]
    total = len(lots)
    lots = lots.sort_values(["fecha_caducidad_lote_proximo", "tienda_id", "sku"])
    return {
        "filters": {
            "within_days": within_days,
            "store_id": store_id,
            "category": category,
            "inventory_snapshot_date": INVENTORY_SNAPSHOT_DATE.isoformat(),
            "through_date": cutoff.isoformat(),
        },
        "total": total,
        "results": _clean_records(lots),
    }


def _filter_tickets(
    start_date: date | str | None,
    end_date: date | str | None,
    store_id: str | None,
    category: str | None,
    priority: str | None,
    state: str | None,
) -> tuple[pd.DataFrame, date | None, date | None]:
    start, end = _validate_ticket_range(start_date, end_date)
    tickets = _load_tickets()
    if start is not None:
        tickets = tickets[tickets["fecha_creacion"].dt.date >= start]
    if end is not None:
        tickets = tickets[tickets["fecha_creacion"].dt.date <= end]
    for column, value in (
        ("tienda_id", store_id),
        ("categoria", category),
        ("prioridad", priority),
        ("estado", state),
    ):
        if value is not None:
            tickets = tickets[tickets[column] == value]
    return tickets, start, end


def ticket_counts(
    group_by: TicketGroup = "category",
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    store_id: str | None = None,
    category: str | None = None,
    priority: str | None = None,
    state: str | None = None,
) -> dict[str, object]:
    """Count tickets by a structured field, optionally applying other filters."""
    columns_by_group = {
        "category": "categoria",
        "priority": "prioridad",
        "state": "estado",
        "store_id": "tienda_id",
    }
    if group_by not in columns_by_group:
        raise ValueError("group_by must be category, priority, state, or store_id")
    group_column = columns_by_group[group_by]
    tickets, start, end = _filter_tickets(
        start_date, end_date, store_id, category, priority, state
    )
    grouped = (
        tickets.groupby(group_column, dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(group_column)
    )
    total = int(grouped["count"].sum())
    return {
        "filters": {
            "group_by": group_by,
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
            "store_id": store_id,
            "category": category,
            "priority": priority,
            "state": state,
        },
        "total": total,
        "results": _clean_records(grouped),
    }


def list_tickets(
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    store_id: str | None = None,
    category: str | None = None,
    priority: str | None = None,
    state: str | None = None,
    limit: int = 20,
) -> dict[str, object]:
    """List recent matching tickets, newest first."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    tickets, start, end = _filter_tickets(
        start_date, end_date, store_id, category, priority, state
    )
    total = len(tickets)
    tickets = tickets.sort_values(["fecha_creacion", "ticket_id"], ascending=False).head(limit)
    return {
        "filters": {
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
            "store_id": store_id,
            "category": category,
            "priority": priority,
            "state": state,
            "limit": limit,
        },
        "total": total,
        "results": _clean_records(tickets),
    }


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _singularize(word: str) -> str:
    # Strips a simple Spanish plural suffix so "cajas" compares equal to "caja".
    if len(word) >= 4:
        if word.endswith("es"):
            return word[:-2]
        if word.endswith("s"):
            return word[:-1]
    return word


def search_tickets(
    text: str,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    store_id: str | None = None,
    category: str | None = None,
    priority: str | None = None,
    state: str | None = None,
    limit: int = 20,
    min_similarity: float = 0.8,
    match_mode: Literal["all", "any"] = "all",
) -> dict[str, object]:
    """Search ticket titles and descriptions with accent-insensitive fuzzy word matching."""
    if not text.strip():
        raise ValueError("text cannot be empty")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between 0 and 1")
    if match_mode not in {"all", "any"}:
        raise ValueError("match_mode must be 'all' or 'any'")
    tickets, start, end = _filter_tickets(
        start_date, end_date, store_id, category, priority, state
    )
    query_tokens = [
        _singularize(token)
        for token in re.findall(r"\w+", _normalize_text(text))
        if token not in SPANISH_STOPWORDS
    ]
    if not query_tokens:
        raise ValueError("text must contain at least one searchable word besides stopwords")

    matches = []
    for record in tickets.to_dict(orient="records"):
        combined_text = f"{record['titulo']} {record['descripcion']}"
        description_tokens = [
            _singularize(token) for token in re.findall(r"\w+", _normalize_text(combined_text))
        ]
        if not description_tokens:
            continue
        scores = []
        for query_token in query_tokens:
            if len(query_token) < 5:
                score = 1.0 if query_token in description_tokens else 0.0
            else:
                score = max(
                    difflib.SequenceMatcher(None, query_token, word).ratio()
                    for word in description_tokens
                )
            scores.append(score)
        matched_words = sum(1 for score in scores if score >= min_similarity)
        is_match = matched_words == len(scores) if match_mode == "all" else matched_words >= 1
        if is_match:
            record["match_similarity"] = round(sum(scores) / len(scores), 4)
            record["_matched_words"] = matched_words
            matches.append(record)
    matches.sort(key=lambda record: (-record["_matched_words"], -record["match_similarity"], -record["ticket_id"]))
    for record in matches:
        del record["_matched_words"]
    total_matches = len(matches)
    results = _clean_records(pd.DataFrame(matches).head(limit)) if matches else []
    return {
        "filters": {
            "text": text,
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
            "store_id": store_id,
            "category": category,
            "priority": priority,
            "state": state,
            "limit": limit,
            "min_similarity": min_similarity,
            "match_mode": match_mode,
        },
        "total_matches": total_matches,
        "results": results,
    }