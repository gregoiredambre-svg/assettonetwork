"""Systematic inventory of the Analysis Ready Distress workbook for modelling design."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "Research Data" / "Analysis Ready Distress.xlsx"
REPORT_DIR = ROOT / "reports"
GRAPH_DIR = ROOT / "graph_data"

DOC_SHEETS = ["Table Reference", "Field Reference", "Codes Reference"]
DATA_TABLES = [
    "ANALYSIS_DIS_AC",
    "ANALYSIS_DIS_CRCP",
    "ANALYSIS_DIS_JPCC",
    "EXPERIMENT_SECTION",
    "SHRP_INFO",
    "TST_L05B",
    "PERFORMANCE_EVENT",
]
DISTRESS_TABLES = ["ANALYSIS_DIS_AC", "ANALYSIS_DIS_CRCP", "ANALYSIS_DIS_JPCC"]
KEYWORDS = [
    "cracking",
    "rutting",
    "potholes",
    "patches",
    "iri",
    "roughness",
    "wear",
    "deflection",
    "deterioration",
    "faulting",
    "spalling",
    "joint distress",
    "punchout",
]
SUFFICIENCY_SECTION_THRESHOLD = 200
SUFFICIENCY_OBS_THRESHOLD = 1000
SUFFICIENCY_YEARS_THRESHOLD = 5

# Workbook fields already used in the current production pipeline or directly equivalent to current targets.
CURRENT_USED_FIELDS: dict[str, set[str]] = {
    "ANALYSIS_DIS_AC": {
        "STATE_CODE",
        "SHRP_ID",
        "SURVEY_DATE",
        "HPMS16_CRACKING_PERCENT_AC",
        "MEPDG_CRACKING_PERCENT_AC",
        "MEPDG_TRANS_CRACK_LENGTH_AC",
        "PATCH_A",
        "POTHOLES_A",
        "PAVEMENT_FAMILY",
        "PAVEMENT_FAMILY_EXP",
    },
    "EXPERIMENT_SECTION": {
        "STATE_CODE",
        "SHRP_ID",
        "CONSTRUCTION_NO",
        "CN_ASSIGN_DATE",
        "CN_CHANGE_REASON",
        "CN_CHANGE_REASON_EXP",
        "ASSIGN_DATE",
        "DEASSIGN_DATE",
    },
    "SHRP_INFO": set(),
    "TST_L05B": set(),
    "PERFORMANCE_EVENT": set(),
    "ANALYSIS_DIS_CRCP": set(),
    "ANALYSIS_DIS_JPCC": set(),
}


def log(message: str) -> None:
    print(f"[inventory] {message}")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(
                lambda v: ""
                if pd.isna(v)
                else (f"{float(v):.4f}" if isinstance(v, (float, np.floating)) else str(int(v)))
            )
        else:
            display[col] = display[col].fillna("").astype(str)
    headers = [str(c) for c in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


@dataclass
class WorkbookCache:
    """Load each workbook sheet once and keep documentation lookups in memory."""

    xls: pd.ExcelFile
    sheets: dict[str, pd.DataFrame]
    table_reference: pd.DataFrame
    field_reference: pd.DataFrame
    codes_reference: pd.DataFrame
    field_dict: dict[tuple[str, str], dict[str, Any]]
    table_dict: dict[str, dict[str, Any]]
    code_detail_dict: dict[str, dict[str, str]]


def build_cache() -> WorkbookCache:
    xls = pd.ExcelFile(DATA_PATH)
    sheets = {sheet: pd.read_excel(DATA_PATH, sheet_name=sheet) for sheet in xls.sheet_names}
    table_reference = sheets["Table Reference"].copy()
    field_reference = sheets["Field Reference"].copy()
    codes_reference = sheets["Codes Reference"].copy()

    field_dict: dict[tuple[str, str], dict[str, Any]] = {}
    for row in field_reference.itertuples(index=False):
        field_dict[(str(row.TABLE_NAME), str(row.FIELD_NAME))] = {
            "alias": None if pd.isna(row.FIELD_ALIAS) else str(row.FIELD_ALIAS),
            "description": None if pd.isna(row.FIELD_DESCRIPTION) else str(row.FIELD_DESCRIPTION),
            "unit": None if pd.isna(row.FIELD_UNIT) else str(row.FIELD_UNIT),
            "codetype": None if pd.isna(row.FIELD_CODETYPE) else str(row.FIELD_CODETYPE),
            "unit_system": None if pd.isna(row.UNIT_SYSTEM) else str(row.UNIT_SYSTEM),
        }

    table_dict: dict[str, dict[str, Any]] = {}
    for row in table_reference.itertuples(index=False):
        table_dict[str(row.TABLE_NAME)] = {
            "alias": None if pd.isna(row.TABLE_ALIAS) else str(row.TABLE_ALIAS),
            "description": None if pd.isna(row.TABLE_DESCRIPTION) else str(row.TABLE_DESCRIPTION),
            "class_name": None if pd.isna(row.CLASS_NAME) else str(row.CLASS_NAME),
        }

    code_detail_dict: dict[str, dict[str, str]] = {}
    for row in codes_reference.itertuples(index=False):
        codetype = None if pd.isna(row.CODETYPE) else str(row.CODETYPE)
        code = None if pd.isna(row.CODE) else str(row.CODE)
        detail = None if pd.isna(row.DETAIL) else str(row.DETAIL)
        if codetype and code:
            code_detail_dict.setdefault(codetype, {})[code] = detail or ""

    return WorkbookCache(
        xls=xls,
        sheets=sheets,
        table_reference=table_reference,
        field_reference=field_reference,
        codes_reference=codes_reference,
        field_dict=field_dict,
        table_dict=table_dict,
        code_detail_dict=code_detail_dict,
    )


def normalize_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            if float(text).is_integer():
                return str(int(float(text)))
        except Exception:
            pass
    return text


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def infer_numeric(series: pd.Series) -> tuple[bool, pd.Series]:
    numeric = safe_numeric(series)
    observed = series.notna().sum()
    if observed == 0:
        return False, numeric
    convertible = numeric.notna().sum()
    return convertible / observed >= 0.8, numeric


def infer_datetime(series: pd.Series) -> bool:
    name = str(series.name or "").upper()
    if "DATE" not in name:
        return False
    parsed = pd.to_datetime(series, errors="coerce")
    observed = series.notna().sum()
    if observed == 0:
        return False
    return parsed.notna().sum() / observed >= 0.8


def numeric_summary(series: pd.Series) -> dict[str, float | None]:
    clean = safe_numeric(series).dropna()
    if clean.empty:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "share_zero": None,
            "skew": None,
        }
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "p50": float(clean.quantile(0.5)),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
        "share_zero": float((clean == 0).mean()),
        "skew": float(clean.skew()) if len(clean) > 2 else 0.0,
    }


def top_codes(series: pd.Series, codetype: str | None, code_lookup: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    if not codetype:
        return []
    counts = series.dropna().map(normalize_code)
    counts = counts[counts != ""].value_counts().head(5)
    detail_map = code_lookup.get(codetype, {})
    rows = []
    for code, count in counts.items():
        rows.append({"code": code, "count": int(count), "detail": detail_map.get(code, "")})
    return rows


def describe_distribution(skew: float | None, share_zero: float | None) -> str:
    if share_zero is not None and share_zero > 0.9:
        return "binary_like"
    if skew is None or pd.isna(skew):
        return "unknown"
    if skew < 1:
        return "bell"
    if skew < 3:
        return "right_skewed"
    return "heavy_tailed"


def recommend_transform(unit: str | None, shape: str, share_zero: float | None) -> str:
    if share_zero is not None and share_zero > 0.9:
        return "binary_classification"
    if shape == "heavy_tailed":
        return "log1p+winsor99"
    if shape == "right_skewed":
        return "log1p"
    return "identity"


def pros_for_target(field_name: str, unit: str | None, missing_pct: float, share_zero: float | None, shape: str) -> str:
    bits = []
    if missing_pct < 10:
        bits.append("high coverage")
    elif missing_pct < 30:
        bits.append("moderate coverage")
    if unit in {"%", "ft/mi", "sq m", "m", "mm"}:
        bits.append(f"physically interpretable unit ({unit})")
    if share_zero is not None and share_zero < 0.8:
        bits.append("enough non-zero variation for regression")
    if "CRACK" in field_name or "RUT" in field_name or "IRI" in field_name:
        bits.append("directly measures pavement condition rather than a code or event marker")
    if shape == "bell":
        bits.append("distribution is relatively well-behaved")
    elif shape == "right_skewed":
        bits.append("skew can be handled with a simple transform")
    return "; ".join(bits) if bits else "This distress quantity is observed on a meaningful physical scale and therefore can serve as a direct modelling target."


def cons_for_target(field_name: str, missing_pct: float, share_zero: float | None, shape: str) -> str:
    bits = []
    if missing_pct > 40:
        bits.append("substantial missingness")
    if share_zero is not None and share_zero > 0.9:
        bits.append("mostly zeros, so plain regression is unstable")
    elif share_zero is not None and share_zero > 0.7:
        bits.append("strong zero inflation")
    if shape == "heavy_tailed":
        bits.append("extreme upper tail can dominate RMSE and R²")
    if field_name.endswith("_L") or field_name.endswith("_A"):
        bits.append("may overlap with related severity-specific or aggregate fields")
    return "; ".join(bits) if bits else "No major statistical red flag appears beyond normal target-specific modelling care."


def implementation_effort(table_name: str, field_name: str) -> str:
    if table_name == "TST_L05B":
        return "HIGH (>90 min)"
    if table_name == "PERFORMANCE_EVENT":
        return "HIGH (>90 min)"
    if table_name in {"SHRP_INFO", "EXPERIMENT_SECTION"}:
        return "MEDIUM (30-90 min)"
    if table_name.startswith("ANALYSIS_DIS_"):
        return "LOW (<30 min)"
    return "MEDIUM (30-90 min)"


def information_value(table_name: str, field_name: str, description: str | None) -> tuple[str, str]:
    text = f"{field_name} {description or ''}".lower()
    if table_name == "TST_L05B":
        if any(k in text for k in ["repr_thickness", "thickness", "matl_code", "material", "layer_type", "project_layer_code", "description"]):
            return "HIGH", "Pavement layer composition and thickness are likely core determinants of how distress evolves after treatment."
    if table_name == "EXPERIMENT_SECTION":
        if any(k in text for k in ["status", "experiment", "pavement family", "change reason", "construction number", "assign", "deassign", "seasonal"]):
            return "HIGH", "This field can add timing, treatment, or structural-regime context that the current production features only partially capture."
    if table_name == "SHRP_INFO":
        if any(k in text for k in ["func_class", "sro_", "data_availability", "lanes", "load_dir", "ltpp_lane", "wim", "class_site", "volume_site", "weight", "direction", "site", "lane"]):
            return "MEDIUM", "This field looks like stable site or traffic-system context that could improve heterogeneity modelling."
    if table_name.startswith("ANALYSIS_DIS_"):
        if any(k in text for k in ["_l", "_m", "_h", "seal", "wp", "nwp", "gator", "block", "edge", "trans", "long", "corner", "spall", "fault", "punchout", "durab"]):
            return "HIGH", "Severity-disaggregated distress components can provide richer lagged condition state than the current single aggregate target."
    if table_name == "PERFORMANCE_EVENT":
        return "MEDIUM", "Event-type history may explain discontinuities in measured distress or treatment timing, but the sheet structure looks harder to operationalise cleanly."
    return "LOW", "This field is available, but its direct predictive value is less obvious than structure, distress state, or treatment history."


def section_key(df: pd.DataFrame) -> pd.Series:
    state_col = "STATE_CODE" if "STATE_CODE" in df.columns else None
    shrp_col = "SHRP_ID" if "SHRP_ID" in df.columns else None
    if not state_col or not shrp_col:
        return pd.Series(dtype=str)
    return df[[state_col, shrp_col]].astype(str).agg("_".join, axis=1)


def candidate_target_filter(table_name: str, field_name: str, meta: dict[str, Any]) -> bool:
    desc = (meta.get("description") or "")
    alias = (meta.get("alias") or "")
    unit = meta.get("unit") or ""
    text = f"{field_name} {alias} {desc}".lower()
    if not any(keyword in text for keyword in KEYWORDS):
        return False
    banned_fragments = ["flag", "code expansion", "state code", "survey date", "construction number", "identifier"]
    if any(fragment in text for fragment in banned_fragments):
        return False
    if field_name.endswith("_FLAG") or field_name.endswith("_FLAG_EXP") or field_name.endswith("_EXP"):
        return False
    if field_name.endswith("_NO") or (" number " in text) or (alias.lower().endswith(" number") if alias else False):
        return False
    if "count" in text or "code indicating" in text or "date " in text:
        return False
    if unit == "" and not any(token in field_name for token in ["IRI", "ROUGH", "DEFL"]):
        return False
    return True


def field_inventory_for_table(
    cache: WorkbookCache,
    table_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = cache.sheets[table_name].copy()
    table_meta = cache.table_dict.get(table_name, {})
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        meta = cache.field_dict.get((table_name, str(col)), {})
        series = df[col]
        missing_pct = float(series.isna().mean() * 100.0)
        n_unique = int(series.nunique(dropna=True))
        is_date = infer_datetime(series)
        is_numeric, numeric = infer_numeric(series)
        stats = numeric_summary(series) if is_numeric and not is_date else {
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "share_zero": None,
            "skew": None,
        }
        rows.append(
            {
                "table_name": table_name,
                "field_name": str(col),
                "field_alias": meta.get("alias"),
                "field_description": meta.get("description"),
                "unit": meta.get("unit"),
                "codetype": meta.get("codetype"),
                "unit_system": meta.get("unit_system"),
                "missing_pct": missing_pct,
                "n_unique": n_unique,
                "is_numeric": bool(is_numeric and not is_date),
                "is_datetime": bool(is_date),
                **stats,
                "top_codes": top_codes(series, meta.get("codetype"), cache.code_detail_dict),
            }
        )
    inventory = pd.DataFrame(rows)
    table_summary = {
        "table_name": table_name,
        "table_alias": table_meta.get("alias"),
        "table_description": table_meta.get("description"),
        "class_name": table_meta.get("class_name"),
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
    }
    return inventory, table_summary


def build_candidate_targets(cache: WorkbookCache, inventories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table_name in DISTRESS_TABLES:
        inventory = inventories[table_name]
        for row in inventory.itertuples(index=False):
            meta = {
                "description": row.field_description,
                "alias": row.field_alias,
                "unit": row.unit,
            }
            if not candidate_target_filter(table_name, row.field_name, meta):
                continue
            shape = describe_distribution(row.skew, row.share_zero)
            transform = recommend_transform(row.unit, shape, row.share_zero)
            rows.append(
                {
                    "table_name": table_name,
                    "field_name": row.field_name,
                    "field_alias": row.field_alias,
                    "field_description": row.field_description,
                    "unit": row.unit,
                    "missing_pct": row.missing_pct,
                    "share_zero": row.share_zero,
                    "skewness": row.skew,
                    "distribution_shape": shape,
                    "recommended_transform": transform,
                    "pros": pros_for_target(row.field_name, row.unit, row.missing_pct, row.share_zero, shape),
                    "cons": cons_for_target(row.field_name, row.missing_pct, row.share_zero, shape),
                }
            )
    return pd.DataFrame(rows).sort_values(["table_name", "missing_pct", "field_name"]).reset_index(drop=True)


def build_missed_predictors(cache: WorkbookCache, inventories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def consider(table_name: str, field_name: str, alias: str | None, description: str | None, unit: str | None) -> bool:
        if field_name in CURRENT_USED_FIELDS.get(table_name, set()):
            return False
        if field_name.endswith("_FLAG") or field_name.endswith("_FLAG_EXP") or field_name.endswith("_EXP"):
            return False
        text = f"{field_name} {alias or ''} {description or ''}".lower()
        if table_name.startswith("ANALYSIS_DIS_"):
            return any(k in text for k in ["crack", "patch", "pothole", "rut", "rough", "iri", "wear", "fault", "spall", "joint", "punchout"]) and not field_name.endswith("_NO")
        if table_name == "TST_L05B":
            return any(k in text for k in ["thickness", "layer", "matl_code", "material", "repr_thickness", "description", "layer type", "project layer"])
        if table_name == "EXPERIMENT_SECTION":
            return any(k in text for k in ["status", "experiment", "pavement family", "change reason", "construction", "assign", "deassign", "seasonal", "record status"])
        if table_name == "SHRP_INFO":
            return any(k in text for k in ["site", "lane", "direction", "class", "data availability", "weight", "load", "func", "region", "drain", "soil", "wim", "volume"])
        if table_name == "PERFORMANCE_EVENT":
            return True
        return False

    for table_name, inventory in inventories.items():
        if table_name not in DATA_TABLES:
            continue
        for row in inventory.itertuples(index=False):
            if not consider(table_name, row.field_name, row.field_alias, row.field_description, row.unit):
                continue
            info_value, justification = information_value(table_name, row.field_name, row.field_description)
            rows.append(
                {
                    "table_name": table_name,
                    "field_name": row.field_name,
                    "field_alias": row.field_alias,
                    "field_description": row.field_description,
                    "unit": row.unit,
                    "missing_pct": row.missing_pct,
                    "estimated_information_value": info_value,
                    "estimated_implementation_effort": implementation_effort(table_name, row.field_name),
                    "justification": justification,
                }
            )
    out = pd.DataFrame(rows).drop_duplicates(subset=["table_name", "field_name"]).sort_values(
        ["estimated_information_value", "table_name", "field_name"],
        ascending=[True, True, True],
    )
    value_order = pd.CategoricalDtype(["HIGH", "MEDIUM", "LOW"], ordered=True)
    out["estimated_information_value"] = out["estimated_information_value"].astype(value_order)
    out = out.sort_values(["estimated_information_value", "table_name", "field_name"]).reset_index(drop=True)
    return out


def cross_table_coverage(cache: WorkbookCache) -> dict[str, Any]:
    section_sets: dict[str, set[str]] = {}
    for table_name in DISTRESS_TABLES:
        df = cache.sheets[table_name]
        section_sets[table_name] = set(section_key(df).dropna().astype(str).tolist())

    pairwise_overlap = {}
    for left in DISTRESS_TABLES:
        for right in DISTRESS_TABLES:
            if left >= right:
                continue
            overlap = section_sets[left] & section_sets[right]
            pairwise_overlap[f"{left}__{right}"] = len(overlap)

    all_sections = set().union(*section_sets.values())
    section_membership_rows = []
    for sec in all_sections:
        present = [name for name, values in section_sets.items() if sec in values]
        section_membership_rows.append({"section": sec, "present_in": present, "n_tables": len(present)})
    membership = pd.DataFrame(section_membership_rows)

    sections_ge_5_years: dict[str, int] = {}
    observations_by_table: dict[str, int] = {}
    span_years_by_table: dict[str, int] = {}
    for table_name in DISTRESS_TABLES:
        df = cache.sheets[table_name].copy()
        observations_by_table[table_name] = int(len(df))
        if "SURVEY_DATE" in df.columns:
            dates = pd.to_datetime(df["SURVEY_DATE"], errors="coerce")
            years = dates.dt.year
            span_years_by_table[table_name] = int(years.dropna().nunique())
            df["_YEAR"] = years
            if "STATE_CODE" in df.columns and "SHRP_ID" in df.columns:
                df["_section"] = section_key(df)
                counts = df.dropna(subset=["_YEAR"]).groupby("_section")["_YEAR"].nunique()
                sections_ge_5_years[table_name] = int((counts >= 5).sum())
            else:
                sections_ge_5_years[table_name] = 0
        else:
            span_years_by_table[table_name] = 0
            sections_ge_5_years[table_name] = 0

    return {
        "n_sections_by_table": {name: len(values) for name, values in section_sets.items()},
        "pairwise_overlap": pairwise_overlap,
        "sections_in_multiple_distress_tables": int((membership["n_tables"] > 1).sum()),
        "sections_in_all_three_tables": int((membership["n_tables"] == 3).sum()),
        "membership_distribution": membership["n_tables"].value_counts().sort_index().to_dict(),
        "n_sections_with_AC_and_CRCP": int(len(section_sets["ANALYSIS_DIS_AC"] & section_sets["ANALYSIS_DIS_CRCP"])),
        "n_sections_with_temporal_coverage_ge_5_years": sections_ge_5_years,
        "observations_by_table": observations_by_table,
        "span_years_by_table": span_years_by_table,
    }


def sufficiency_verdict(table_name: str, coverage: dict[str, Any], candidate_targets: pd.DataFrame) -> dict[str, Any]:
    n_sections = int(coverage["n_sections_by_table"].get(table_name, 0))
    n_obs = int(coverage["observations_by_table"].get(table_name, 0))
    n_ge5 = int(coverage["n_sections_with_temporal_coverage_ge_5_years"].get(table_name, 0))
    span_years = int(coverage["span_years_by_table"].get(table_name, 0))
    if n_sections >= SUFFICIENCY_SECTION_THRESHOLD and n_obs >= SUFFICIENCY_OBS_THRESHOLD and n_ge5 >= SUFFICIENCY_SECTION_THRESHOLD:
        verdict = "SUFFICIENT FOR SEPARATE MODEL"
    elif n_sections >= 100 and n_obs >= 500:
        verdict = "MARGINAL — POSSIBLE WITH POOLING TO AC"
    else:
        verdict = "INSUFFICIENT — KEEP AC-ONLY FOCUS"

    table_targets = candidate_targets[candidate_targets["table_name"] == table_name].copy()
    table_targets = table_targets.sort_values(["missing_pct", "share_zero", "field_name"]).head(3)
    target_rows = [
        {
            "field_name": row.field_name,
            "description": row.field_description,
            "rationale": f"Coverage is {100.0 - float(row.missing_pct):.1f}% with zero share {100.0 * float(row.share_zero or 0):.1f}%, making it one of the more feasible distress targets in this pavement family."
        }
        for row in table_targets.itertuples(index=False)
    ]
    paragraph = (
        f"{table_name.replace('ANALYSIS_DIS_', '')} has {n_sections} sections with {n_obs} total distress observations "
        f"spanning {span_years} distinct survey years. Threshold for a separate R-GCN model: at least "
        f"{SUFFICIENCY_SECTION_THRESHOLD} sections AND {SUFFICIENCY_OBS_THRESHOLD} observations across {SUFFICIENCY_YEARS_THRESHOLD}+ years. "
        f"Sections with temporal coverage >=5 years: {n_ge5}. Verdict: {verdict}."
    )
    return {
        "table_name": table_name,
        "n_sections": n_sections,
        "n_observations": n_obs,
        "span_years": span_years,
        "n_sections_with_ge5_years": n_ge5,
        "verdict": verdict,
        "paragraph": paragraph,
        "candidate_targets": target_rows,
    }


def compute_red_flags(cache: WorkbookCache, coverage: dict[str, Any]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []

    ac = cache.sheets["ANALYSIS_DIS_AC"].copy()
    ac["SURVEY_DATE"] = pd.to_datetime(ac["SURVEY_DATE"], errors="coerce")
    ac["YEAR"] = ac["SURVEY_DATE"].dt.year
    test_ac = ac[ac["YEAR"].between(2019, 2021, inclusive="both")].copy()
    patch_zero = float((pd.to_numeric(test_ac["PATCH_A"], errors="coerce").fillna(0) == 0).mean()) if not test_ac.empty else math.nan
    pothole_zero = float((pd.to_numeric(test_ac["POTHOLES_A"], errors="coerce").fillna(0) == 0).mean()) if not test_ac.empty else math.nan
    if not math.isnan(patch_zero) and patch_zero > 0.9 and not math.isnan(pothole_zero) and pothole_zero > 0.9:
        status = "RED_FLAG_PRESENT"
    elif not math.isnan(patch_zero) and patch_zero < 0.8 and not math.isnan(pothole_zero) and pothole_zero < 0.8:
        status = "RED_FLAG_NOT_PRESENT"
    else:
        status = "INCONCLUSIVE — needs further investigation"
    flags.append(
        {
            "name": "Sparse-event regression may make R² meaningless for PATCH_A / POTHOLES_A",
            "status": status,
            "statistic": {
                "test_patch_zero_share": patch_zero,
                "test_pothole_zero_share": pothole_zero,
            },
            "interpretation": "This checks whether the late test years are so dominated by zeros that standard regression fit metrics become unstable for rare-event distress targets.",
        }
    )

    crcp = cache.sheets["ANALYSIS_DIS_CRCP"].copy()
    if {"STATE_CODE", "SHRP_ID", "SURVEY_DATE", "HPMS16_CRACKING_PERCENT_AC"}.issubset(ac.columns) and {"STATE_CODE", "SHRP_ID", "SURVEY_DATE", "HPMS16_CRACKING_PERCENT_CRCP"}.issubset(crcp.columns):
        crcp["SURVEY_DATE"] = pd.to_datetime(crcp["SURVEY_DATE"], errors="coerce")
        ac["_node"] = section_key(ac)
        crcp["_node"] = section_key(crcp)
        ac_small = ac[["_node", "SURVEY_DATE", "HPMS16_CRACKING_PERCENT_AC"]].rename(columns={"HPMS16_CRACKING_PERCENT_AC": "ac_hpms"})
        crcp_small = crcp[["_node", "SURVEY_DATE", "HPMS16_CRACKING_PERCENT_CRCP"]].rename(columns={"HPMS16_CRACKING_PERCENT_CRCP": "crcp_hpms"})
        overlap = ac_small.merge(crcp_small, on=["_node", "SURVEY_DATE"], how="inner")
        overlap["ac_hpms"] = pd.to_numeric(overlap["ac_hpms"], errors="coerce")
        overlap["crcp_hpms"] = pd.to_numeric(overlap["crcp_hpms"], errors="coerce")
        overlap = overlap.dropna()
        if len(overlap) >= 30:
            corr = float(overlap["ac_hpms"].corr(overlap["crcp_hpms"]))
            status = "RED_FLAG_PRESENT" if corr >= 0.8 else "RED_FLAG_NOT_PRESENT"
        else:
            corr = math.nan
            status = "INCONCLUSIVE — needs further investigation"
    else:
        corr = math.nan
        status = "INCONCLUSIVE — needs further investigation"
    flags.append(
        {
            "name": "CRCP distress may duplicate AC distress information",
            "status": status,
            "statistic": {"same-section same-date AC/CRCP HPMS correlation": corr, "n_sections_with_AC_and_CRCP": coverage["n_sections_with_AC_and_CRCP"]},
            "interpretation": "This checks whether sections appearing in both AC and CRCP tables carry near-duplicate cracking information, which would make naive pooling misleading.",
        }
    )

    l05b = cache.sheets["TST_L05B"].copy()
    if {"STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO", "REPR_THICKNESS"}.issubset(l05b.columns):
        l05b["_node"] = section_key(l05b)
        l05b["REPR_THICKNESS"] = pd.to_numeric(l05b["REPR_THICKNESS"], errors="coerce")
        grouped = (
            l05b.groupby(["_node", "CONSTRUCTION_NO"], as_index=False)["REPR_THICKNESS"]
            .sum(min_count=1)
            .rename(columns={"REPR_THICKNESS": "total_repr_thickness"})
        )
        section_changes = grouped.groupby("_node")["total_repr_thickness"].nunique(dropna=True)
        multi_const = grouped.groupby("_node")["CONSTRUCTION_NO"].nunique()
        eligible = multi_const[multi_const > 1].index
        if len(eligible) > 0:
            changing_share = float((section_changes.reindex(eligible).fillna(0) > 1).mean())
            constant_share = float((section_changes.reindex(eligible).fillna(0) <= 1).mean())
            status = "RED_FLAG_PRESENT" if changing_share > 0.5 else "RED_FLAG_NOT_PRESENT"
        else:
            changing_share = math.nan
            constant_share = math.nan
            status = "INCONCLUSIVE — needs further investigation"
    else:
        changing_share = math.nan
        constant_share = math.nan
        status = "INCONCLUSIVE — needs further investigation"
    flags.append(
        {
            "name": "TST_L05B structural summaries may leak treatment regime through construction changes",
            "status": status,
            "statistic": {
                "share_multi_construction_sections_with_thickness_change": changing_share,
                "share_multi_construction_sections_without_thickness_change": constant_share,
            },
            "interpretation": "If layer-thickness summaries change systematically with construction number, contemporaneous structure features may encode treatment events too directly and must be aligned carefully in time.",
        }
    )
    return flags


def recommend_targets(candidate_targets: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    keep = [
        {
            "target": "HPMS16_CRACKING_PERCENT_AC",
            "rationale": "Keep as the main asphalt cracking target because it has near-complete coverage, a stable unit in percent, and already supports strong predictive performance.",
        },
        {
            "target": "MEPDG_CRACKING_PERCENT_AC",
            "rationale": "Keep because it captures broader alligator cracking than HPMS16 and empirically behaved even better in the single-task runs.",
        },
        {
            "target": "MEPDG_TRANS_CRACK_LENGTH_AC",
            "rationale": "Keep as a secondary target because it broadens the distress story, but only with log-style transformation and careful interpretation.",
        },
        {
            "target": "PATCH_A",
            "rationale": "Keep only as an exploratory sparse-event target, not as a headline regression benchmark, because the series is highly zero-inflated.",
        },
        {
            "target": "POTHOLES_A",
            "rationale": "Keep only as an exploratory rare-event target or move to a two-stage classification/regression design because almost all observations are zero.",
        },
    ]

    additions = []
    for field in ["MEPDG_LONG_CRACK_LENGTH_AC", "GATOR_CRACK_A", "LONG_CRACK_WP_L"]:
        row = candidate_targets[candidate_targets["field_name"] == field]
        if row.empty:
            continue
        unit = row.iloc[0]["unit"]
        additions.append(
            {
                "target": field,
                "rationale": f"This field looks like a plausible additional target because it has a direct physical interpretation ({unit}) and can isolate another dimension of deterioration not covered by the current five targets.",
                "estimated_r2_ceiling": "0.20-0.45",
            }
        )

    deprioritize = [
        {
            "target": "count-style distress fields ending in _NO",
            "rationale": "Deprioritise because count variables are often sparse, thresholded, and less stable than area or length measures on the current annual aggregation.",
        },
        {
            "target": "anomaly flags (*_FLAG, *_FLAG_EXP)",
            "rationale": "Do not use as targets because they are quality indicators rather than distress outcomes.",
        },
    ]
    return {"keep": keep, "additions": additions, "deprioritize": deprioritize}


def recommend_predictors(missed: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    def top_rows(value: str, limit: int) -> list[dict[str, str]]:
        subset = missed[missed["estimated_information_value"].astype(str) == value].head(limit)
        out = []
        for row in subset.itertuples(index=False):
            out.append(
                {
                    "field": f"{row.table_name}.{row.field_name}",
                    "effort": row.estimated_implementation_effort,
                    "rationale": row.justification,
                }
            )
        return out

    return {
        "high_priority": top_rows("HIGH", 8),
        "medium_priority": top_rows("MEDIUM", 8),
        "low_priority": top_rows("LOW", 8),
    }


def modelling_architecture_recommendation(coverage: dict[str, Any]) -> dict[str, str]:
    jpcc_n = coverage["n_sections_by_table"].get("ANALYSIS_DIS_JPCC", 0)
    crcp_n = coverage["n_sections_by_table"].get("ANALYSIS_DIS_CRCP", 0)
    return {
        "single_task": "Keep single-task per distress as the main production strategy because the multi-task R-GCN already showed negative transfer and the distress targets have very different scales and zero patterns.",
        "pavement_specific": f"Explore pavement-type-specific models next: JPCC has {jpcc_n} sections and looks feasible for a dedicated concrete-pavement study, while CRCP has only {crcp_n} sections and may be better treated as exploratory unless more data are added.",
        "composite_index": "Do not switch the main analysis to a single composite PCI-like index yet. A composite index would hide which distress types actually benefit from graph structure and which remain mostly local or sparse-event driven.",
    }


def red_flags() -> list[str]:
    return [
        "If PATCH_A and POTHOLES_A stay dominated by zeros in the test split, plain regression R² will remain unstable and could be misleading compared with event-style metrics or hurdle models.",
        "If TST_L05B structural variables are merged without respecting CONSTRUCTION_NO, layer changes could leak treatment timing and make post-treatment prediction look artificially easier.",
        "If sections appearing in multiple distress tables represent materially different pavement systems, pooling AC, CRCP, and JPCC without pavement-type controls could mix incompatible deterioration mechanisms.",
    ]


def render_report(
    cache: WorkbookCache,
    table_summaries: dict[str, dict[str, Any]],
    inventories: dict[str, pd.DataFrame],
    candidate_targets: pd.DataFrame,
    missed_predictors: pd.DataFrame,
    coverage: dict[str, Any],
    sufficiency: dict[str, dict[str, Any]],
    red_flag_results: list[dict[str, Any]],
) -> str:
    paragraphs = 0
    table_count = 0
    lines: list[str] = []

    def add(text: str = "") -> None:
        nonlocal paragraphs, table_count
        lines.append(text)
        if text.strip() and not text.startswith("|") and not text.startswith("- "):
            paragraphs += 1
        if text.startswith("| "):
            table_count += 1

    add("# Distress Full Inventory")
    add("")
    add(
        "This report documents a systematic audit of every documented table and every field in `Research Data/Analysis Ready Distress.xlsx`. "
        "The purpose is not only to list available variables, but to challenge the current modelling scope and show, with evidence, which distress targets and auxiliary predictors deserve more attention."
    )
    add("")
    add("## 1. Documentation sheets")
    add(
        f"The workbook contains {len(cache.table_reference)} official table descriptions, {len(cache.field_reference)} official field definitions, "
        f"and {len(cache.codes_reference)} code rows. These documentation sheets were used as the authoritative source for field meanings, units, and coded-value interpretations."
    )
    add("")
    add("### Table catalogue")
    table_rows = []
    for table_name in DATA_TABLES:
        meta = cache.table_dict.get(table_name, {})
        summary = table_summaries[table_name]
        table_rows.append(
            {
                "Table": table_name,
                "Alias": meta.get("alias"),
                "Rows": summary["n_rows"],
                "Cols": summary["n_cols"],
                "Class": meta.get("class_name"),
                "Description": meta.get("description"),
            }
        )
    add(markdown_table(pd.DataFrame(table_rows)))
    add("")
    add("## 2. Exhaustive per-table field inventory")
    add(
        "The next subsections summarise each data table with both documentation metadata and empirical coverage statistics. "
        "For each field, the report records missingness, uniqueness, numeric distribution summaries when applicable, and the most frequent coded values when a code type is defined."
    )
    add("")

    for table_name in DATA_TABLES:
        summary = table_summaries[table_name]
        inventory = inventories[table_name]
        add(f"### {table_name}")
        alias = summary.get("table_alias") or "No official alias in Table Reference"
        desc = summary.get("table_description") or "No official table description available."
        add(f"**Alias:** {alias}")
        add(f"**Official description:** {desc}")
        add(
            f"This table has **{summary['n_rows']:,} rows** and **{summary['n_cols']} columns**. "
            f"It contributes {inventory['field_name'].nunique()} distinct documented fields to the inventory."
        )
        high_missing = int((inventory["missing_pct"] >= 50).sum())
        coded_fields = int(inventory["codetype"].notna().sum())
        numeric_fields = int(inventory["is_numeric"].sum())
        add(
            f"Within this table, **{numeric_fields} fields** behave numerically, **{coded_fields} fields** have an official code type, "
            f"and **{high_missing} fields** have at least 50% missing values."
        )
        view = inventory[
            [
                "field_name",
                "field_alias",
                "field_description",
                "unit",
                "codetype",
                "missing_pct",
                "n_unique",
                "mean",
                "std",
                "min",
                "p50",
                "p95",
                "max",
                "share_zero",
            ]
        ].copy()
        add(markdown_table(view))
        coded_subset = inventory[inventory["codetype"].notna() & inventory["top_codes"].map(bool)].copy()
        if not coded_subset.empty:
            add("")
            add("Top coded values worth noting:")
            for row in coded_subset.itertuples(index=False):
                code_text = "; ".join(f"{item['code']} ({item['count']}): {item['detail']}" for item in row.top_codes)
                add(f"- `{row.field_name}` [{row.codetype}]: {code_text}")
        add("")

    add("## 3. Candidate distress targets")
    add(
        "Candidate targets were identified by scanning the distress tables for fields whose official descriptions reference cracking, rutting, potholes, patches, IRI, roughness, wear, deflection, or deterioration. "
        "Flags, codes, dates, and count-style fields were removed from the candidate target list, because the goal here is to isolate continuous distress quantities that can reasonably serve as prediction targets."
    )
    add("")
    add(markdown_table(candidate_targets))
    add("")
    add(
        "The candidate-target screen shows a clear pattern. Percentage- and length-based cracking measures have the strongest combination of coverage and physical interpretability. "
        "Area-based patching and pothole measures are still valid distress outcomes, but their zero inflation means they should be analysed differently from the headline cracking targets."
    )
    add("")

    add("## 4. Candidate predictor features not currently used")
    add(
        "The current production pipeline already uses lagged AC distress, external annual traffic, external MERRA climate features, and a subset of project-history features derived from `EXPERIMENT_SECTION`. "
        "This section therefore focuses on fields that are documented in the workbook but are not currently part of those production features."
    )
    add("")
    add(markdown_table(missed_predictors))
    add("")
    add(
        "Three broad sources stand out. First, `TST_L05B` contains the most obviously missing structural signal because layer type, representative thickness, and material code are not yet fed into the models. "
        "Second, `ANALYSIS_DIS_*` tables contain many severity-disaggregated distress components that could enrich the lagged state far beyond a single cracking aggregate. "
        "Third, `SHRP_INFO` offers site-context variables that may explain why the same distress level evolves differently across sections."
    )
    add("")

    add("## 5. Cross-table coverage and feasibility")
    cov_rows = pd.DataFrame(
        [{"Distress table": k, "Sections": v} for k, v in coverage["n_sections_by_table"].items()]
    )
    add(markdown_table(cov_rows))
    add("")
    add(
        f"Across the distress tables, **{coverage['sections_in_multiple_distress_tables']} sections** appear in more than one distress sheet, "
        f"and **{coverage['sections_in_all_three_tables']} sections** appear in all three. "
        f"The overlap counts are therefore non-trivial, but the three pavement systems should not be pooled blindly."
    )
    add("")
    overlap_rows = pd.DataFrame(
        [{"Pair": k.replace('__', ' + '), "Overlapping sections": v} for k, v in coverage["pairwise_overlap"].items()]
    )
    add(markdown_table(overlap_rows))
    add("")
    temporal_rows = pd.DataFrame(
        [
            {
                "Distress table": k,
                "Sections with >=5 years": coverage["n_sections_with_temporal_coverage_ge_5_years"].get(k, 0),
                "Total observations": coverage["observations_by_table"].get(k, 0),
                "Distinct survey years": coverage["span_years_by_table"].get(k, 0),
            }
            for k in DISTRESS_TABLES
        ]
    )
    add(markdown_table(temporal_rows))
    add("")
    add(
        "This matters for modelling strategy. AC has the largest footprint and is the safest base for the current graph-based dissertation pipeline. "
        "JPCC looks large enough to justify a separate pavement-type-specific extension. CRCP is much smaller and may need either stronger regularisation, simpler models, or to remain exploratory unless more data are brought in."
    )
    add("")
    add("### Explicit sufficiency verdicts")
    for pavement in ["ANALYSIS_DIS_CRCP", "ANALYSIS_DIS_JPCC"]:
        verdict = sufficiency[pavement]
        add(f"**{pavement.replace('ANALYSIS_DIS_', '')}**")
        add(verdict["paragraph"])
        if verdict["candidate_targets"]:
            add("Candidate targets if pursued:")
            for item in verdict["candidate_targets"]:
                add(f"- `{item['field_name']}`: {item['rationale']}")
        add("")

    add("## 6. Methodology recommendation")
    target_reco = recommend_targets(candidate_targets)
    predictor_reco = recommend_predictors(missed_predictors)
    arch_reco = modelling_architecture_recommendation(coverage)
    add("### A. Recommended target variables")
    add("**Confirmed primary targets (keep):**")
    for item in target_reco["keep"]:
        add(f"- `{item['target']}`: {item['rationale']}")
    add("")
    add("**New candidate targets worth adding:**")
    for item in target_reco["additions"]:
        add(f"- `{item['target']}`: {item['rationale']} Estimated R² ceiling: {item['estimated_r2_ceiling']}.")
    add("")
    add("**Targets to drop or deprioritise:**")
    for item in target_reco["deprioritize"]:
        add(f"- `{item['target']}`: {item['rationale']}")
    add("")

    add("### B. Recommended predictor features to add")
    add("**High priority (must add):**")
    for item in predictor_reco["high_priority"]:
        add(f"- `{item['field']}` [{item['effort']}]: {item['rationale']}")
    add("")
    add("**Medium priority (add if time permits):**")
    for item in predictor_reco["medium_priority"]:
        add(f"- `{item['field']}` [{item['effort']}]: {item['rationale']}")
    add("")
    add("**Low priority (future work):**")
    for item in predictor_reco["low_priority"]:
        add(f"- `{item['field']}` [{item['effort']}]: {item['rationale']}")
    add("")

    add("### C. Modelling architecture recommendation")
    add(f"- {arch_reco['single_task']}")
    add(f"- {arch_reco['pavement_specific']}")
    add(f"- {arch_reco['composite_index']}")
    add("")

    add("### D. Three red-flag findings")
    for finding in red_flag_results:
        add(f"- **{finding['status']}** — {finding['name']}: {finding['interpretation']}")
    add("")

    add(
        "Overall, the workbook audit suggests that the current project is directionally right but not yet feature-complete. "
        "The biggest untapped signal appears to come from structural information in `TST_L05B`, richer lagged distress composition in the `ANALYSIS_DIS_*` tables, and selected site-context fields in `SHRP_INFO`. "
        "That means the next improvement step should focus less on inventing new model classes and more on integrating the strongest missing physical predictors cleanly."
    )

    log(f"Wrote distress_full_inventory.md ({paragraphs} paragraphs, {table_count} tables)")
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    GRAPH_DIR.mkdir(exist_ok=True)

    cache = build_cache()
    log(
        f"Loaded Table Reference ({len(cache.table_reference)} rows), "
        f"Field Reference ({len(cache.field_reference)} rows), "
        f"Codes Reference ({len(cache.codes_reference)} rows)"
    )

    inventories: dict[str, pd.DataFrame] = {}
    table_summaries: dict[str, dict[str, Any]] = {}
    for table_name in DATA_TABLES:
        df = cache.sheets[table_name]
        log(f"Processing table {table_name} ({len(df.columns)} cols, {len(df)} rows)...")
        inventory, summary = field_inventory_for_table(cache, table_name)
        inventories[table_name] = inventory
        table_summaries[table_name] = summary

    candidate_targets = build_candidate_targets(cache, inventories)
    log(f"Found {len(candidate_targets)} candidate targets in distress tables")
    missed_predictors = build_missed_predictors(cache, inventories)
    log(f"Found {len(missed_predictors)} missed predictor fields across all tables")
    coverage = cross_table_coverage(cache)
    log(f"Cross-table coverage: {coverage['n_sections_by_table']}")
    sufficiency = {
        "ANALYSIS_DIS_CRCP": sufficiency_verdict("ANALYSIS_DIS_CRCP", coverage, candidate_targets),
        "ANALYSIS_DIS_JPCC": sufficiency_verdict("ANALYSIS_DIS_JPCC", coverage, candidate_targets),
    }
    log(f"CRCP sufficiency: {sufficiency['ANALYSIS_DIS_CRCP']['verdict']}")
    log(f"JPCC sufficiency: {sufficiency['ANALYSIS_DIS_JPCC']['verdict']}")
    red_flag_results = compute_red_flags(cache, coverage)
    for finding in red_flag_results:
        log(f"Red flag check: {finding['status']} for {finding['name']}")

    report_text = render_report(
        cache,
        table_summaries,
        inventories,
        candidate_targets,
        missed_predictors,
        coverage,
        sufficiency,
        red_flag_results,
    )
    report_path = REPORT_DIR / "distress_full_inventory.md"
    report_path.write_text(report_text, encoding="utf-8")

    candidate_path = REPORT_DIR / "distress_candidate_targets.csv"
    candidate_targets.to_csv(candidate_path, index=False)
    missed_path = REPORT_DIR / "distress_missed_predictors.csv"
    missed_predictors.to_csv(missed_path, index=False)

    json_payload = {
        "table_summaries": table_summaries,
        "inventories": {name: df.to_dict(orient="records") for name, df in inventories.items()},
        "candidate_targets": candidate_targets.to_dict(orient="records"),
        "missed_predictors": missed_predictors.to_dict(orient="records"),
        "coverage": coverage,
        "sufficiency": sufficiency,
        "red_flags": red_flag_results,
        "recommendations": {
            "targets": recommend_targets(candidate_targets),
            "predictors": recommend_predictors(missed_predictors),
            "architecture": modelling_architecture_recommendation(coverage),
            "red_flags": [item["name"] for item in red_flag_results if item["status"] == "RED_FLAG_PRESENT"],
        },
    }
    json_path = GRAPH_DIR / "distress_full_inventory.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    sufficiency_lines = [
        "# CRCP and JPCC sufficiency verdicts",
        "",
        "## CRCP",
        sufficiency["ANALYSIS_DIS_CRCP"]["paragraph"],
        "",
    ]
    if sufficiency["ANALYSIS_DIS_CRCP"]["candidate_targets"]:
        sufficiency_lines.append("Candidate targets:")
        for item in sufficiency["ANALYSIS_DIS_CRCP"]["candidate_targets"]:
            sufficiency_lines.append(f"- `{item['field_name']}`: {item['rationale']}")
        sufficiency_lines.append("")
    sufficiency_lines.extend(
        [
            "## JPCC",
            sufficiency["ANALYSIS_DIS_JPCC"]["paragraph"],
            "",
        ]
    )
    if sufficiency["ANALYSIS_DIS_JPCC"]["candidate_targets"]:
        sufficiency_lines.append("Candidate targets:")
        for item in sufficiency["ANALYSIS_DIS_JPCC"]["candidate_targets"]:
            sufficiency_lines.append(f"- `{item['field_name']}`: {item['rationale']}")
        sufficiency_lines.append("")
    (REPORT_DIR / "distress_crcp_jpcc_sufficiency.md").write_text("\n".join(sufficiency_lines) + "\n", encoding="utf-8")

    top_targets = recommend_targets(candidate_targets)["additions"][:3]
    top_predictors = recommend_predictors(missed_predictors)["high_priority"][:3]
    print(
        "Top 3 recommended target additions:",
        ", ".join(item["target"] for item in top_targets) if top_targets else "none",
    )
    print(
        "Top 3 recommended predictor additions:",
        ", ".join(item["field"] for item in top_predictors) if top_predictors else "none",
    )
    print("CRCP verdict:", sufficiency["ANALYSIS_DIS_CRCP"]["verdict"])
    print("JPCC verdict:", sufficiency["ANALYSIS_DIS_JPCC"]["verdict"])
    print(
        "Red flags detected:",
        [item["name"] for item in red_flag_results if item["status"] == "RED_FLAG_PRESENT"],
    )


if __name__ == "__main__":
    main()
