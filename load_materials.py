"""Load LTPP TST_L05B layer materials with anti-leakage first-construction snapshots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Research Data" / "Analysis Ready Distress.xlsx"
OUT = ROOT / "graph_data" / "section_materials.csv"
INCH_TO_MM = 25.4

AC_KEYWORDS = ["ASPHALT", "BITUMINOUS", "HOT MIX", "HMA", "HMAC", "AC LAYER"]
PCC_KEYWORDS = ["PCC", "PORTLAND", "CEMENT CONCRETE", "JOINTED PLAIN CONCRETE", "CONTINUOUSLY REINFORCED CONCRETE"]
GRANULAR_KEYWORDS = ["GRANULAR", "GRAVEL", "CRUSHED STONE", "CRUSHED GRAVEL", "SOIL-AGGREGATE"]
BOUND_KEYWORDS = ["CEMENT AGGREGATE", "TREATED", "STABILIZED", "BOUND"]
SURFACE_KEYWORDS = ["SURFACE", "FRICTION COURSE", "SEAL COAT", "SURFACE TREATMENT"]
BINDER_KEYWORDS = ["BINDER"]
BASE_KEYWORDS = ["BASE"]
SUBBASE_KEYWORDS = ["SUBBASE"]
SUBGRADE_KEYWORDS = ["SUBGRADE"]
FABRIC_KEYWORDS = ["ENGINEERING FABRIC", "GEOTEXTILE"]


def normalize_shrp_id(value: object) -> str | None:
    """Normalize SHRP IDs to the same form used by graph construction."""

    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    stripped = text.lstrip("0")
    return stripped or "0"


def build_node_id_join(state_code: pd.Series, shrp_id: pd.Series) -> pd.Series:
    """Build the section join key shared with graph_construction.py."""

    return state_code.astype(str).str.strip() + "_" + shrp_id.map(normalize_shrp_id).fillna("")


def first_construction_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only the earliest available construction snapshot per section."""

    work = raw.copy()
    work["construction_no_num"] = pd.to_numeric(work["CONSTRUCTION_NO"], errors="coerce")
    min_construction = work.groupby("node_id_join")["construction_no_num"].transform("min")
    return work[work["construction_no_num"].eq(min_construction)].copy()


def has_ac_material(series: pd.Series) -> int:
    """Return 1 if any initial-construction layer is clearly asphalt-based."""

    for value in series.dropna():
        text = str(value).upper()
        if any(keyword in text for keyword in AC_KEYWORDS):
            return 1
    return 0


def has_pcc_material(series: pd.Series) -> int:
    """Return 1 if any initial-construction layer is clearly PCC-based.

    Asphalt concrete layers should count as AC, not PCC, even though the word
    "concrete" appears in their textual description.
    """

    for value in series.dropna():
        text = str(value).upper()
        if "ASPHALT" in text or "BITUMINOUS" in text:
            continue
        if any(keyword in text for keyword in PCC_KEYWORDS):
            return 1
        if "CONCRETE" in text and "ASPHALT" not in text:
            return 1
    return 0


def combine_material_text(row: pd.Series) -> str:
    """Combine the material descriptors available in TST_L05B."""

    parts = []
    for col in ["MATL_CODE_EXP", "DESCRIPTION_EXP", "LAYER_TYPE_EXP", "MATL_CODE", "DESCRIPTION", "LAYER_TYPE"]:
        value = row.get(col)
        if pd.notna(value):
            text = str(value).strip()
            if text and text.lower() != "nan":
                parts.append(text)
    return " | ".join(parts)


def text_contains_any(value: object, keywords: list[str]) -> bool:
    """Return True when the given textual value matches one of the keywords."""

    if pd.isna(value):
        return False
    text = str(value).upper()
    return any(keyword in text for keyword in keywords)


def classify_layer_role(row: pd.Series) -> str:
    """Classify a layer into a broad structural role."""

    parts = [
        row.get("DESCRIPTION_EXP"),
        row.get("LAYER_TYPE_EXP"),
        row.get("PROJECT_LAYER_CODE"),
        row.get("DESCRIPTION"),
        row.get("LAYER_TYPE"),
    ]
    text = " | ".join(str(part).upper() for part in parts if pd.notna(part))
    if any(keyword in text for keyword in SURFACE_KEYWORDS):
        return "surface"
    if any(keyword in text for keyword in BINDER_KEYWORDS):
        return "binder"
    if any(keyword in text for keyword in SUBBASE_KEYWORDS):
        return "subbase"
    if any(keyword in text for keyword in BASE_KEYWORDS):
        return "base"
    if any(keyword in text for keyword in SUBGRADE_KEYWORDS):
        return "subgrade"
    if any(keyword in text for keyword in FABRIC_KEYWORDS):
        return "fabric"
    return "other"


def classify_material_family(text: object) -> str:
    """Map raw LTPP material labels to coarse pavement families."""

    if pd.isna(text):
        return "unknown"
    upper = str(text).upper()
    if any(keyword in upper for keyword in AC_KEYWORDS):
        return "ac"
    if ("ASPHALT" not in upper and "BITUMINOUS" not in upper) and (
        any(keyword in upper for keyword in PCC_KEYWORDS) or "CONCRETE" in upper
    ):
        return "pcc"
    if any(keyword in upper for keyword in GRANULAR_KEYWORDS):
        return "granular"
    if any(keyword in upper for keyword in FABRIC_KEYWORDS):
        return "fabric"
    if any(keyword in upper for keyword in BOUND_KEYWORDS):
        return "treated"
    if "SOIL" in upper or "SUBGRADE" in upper:
        return "soil"
    return "other"


def sum_role_thickness(group: pd.DataFrame, role: str) -> float:
    """Aggregate thickness for one structural role within a section."""

    return float(group.loc[group["layer_role"] == role, "REPR_THICKNESS"].fillna(0.0).sum())


def any_role_material(group: pd.DataFrame, role: str, family: str) -> int:
    """Return 1 if a given role contains at least one layer of the chosen family."""

    mask = (group["layer_role"] == role) & (group["material_family"] == family)
    return int(mask.any())


def load_materials() -> pd.DataFrame:
    """Load one anti-leakage material summary row per section.

    The TST_L05B sheet stores representative layer thickness in inches. This loader
    converts aggregate thicknesses to millimetres for downstream similarity use.
    Only the first construction snapshot (or earliest available construction number
    when 1 is absent) is retained, to avoid leaking later treatment events.
    """

    raw = pd.read_excel(DATA, sheet_name="TST_L05B")
    raw["node_id_join"] = build_node_id_join(raw["STATE_CODE"], raw["SHRP_ID"])
    raw = raw.dropna(subset=["node_id_join"]).copy()
    raw["LAYER_NO"] = pd.to_numeric(raw["LAYER_NO"], errors="coerce")
    raw["REPR_THICKNESS"] = pd.to_numeric(raw["REPR_THICKNESS"], errors="coerce")
    raw["material_text"] = raw.apply(combine_material_text, axis=1)
    raw["layer_role"] = raw.apply(classify_layer_role, axis=1)
    raw["material_family"] = raw["material_text"].map(classify_material_family)

    initial = first_construction_snapshot(raw)

    agg = (
        initial.groupby("node_id_join", as_index=False)
        .agg(
            construction_no_snapshot=("construction_no_num", "min"),
            n_layers=("LAYER_NO", "nunique"),
            total_thickness_in=("REPR_THICKNESS", "sum"),
            max_layer_thickness_in=("REPR_THICKNESS", "max"),
            surface_thickness_in=("REPR_THICKNESS", lambda s: float(initial.loc[s.index, "REPR_THICKNESS"][initial.loc[s.index, "layer_role"] == "surface"].fillna(0.0).sum())),
            binder_thickness_in=("REPR_THICKNESS", lambda s: float(initial.loc[s.index, "REPR_THICKNESS"][initial.loc[s.index, "layer_role"] == "binder"].fillna(0.0).sum())),
            base_thickness_in=("REPR_THICKNESS", lambda s: float(initial.loc[s.index, "REPR_THICKNESS"][initial.loc[s.index, "layer_role"] == "base"].fillna(0.0).sum())),
            subbase_thickness_in=("REPR_THICKNESS", lambda s: float(initial.loc[s.index, "REPR_THICKNESS"][initial.loc[s.index, "layer_role"] == "subbase"].fillna(0.0).sum())),
            subgrade_thickness_in=("REPR_THICKNESS", lambda s: float(initial.loc[s.index, "REPR_THICKNESS"][initial.loc[s.index, "layer_role"] == "subgrade"].fillna(0.0).sum())),
            material_codes=("MATL_CODE", lambda x: ",".join(sorted({str(v).strip() for v in x.dropna() if str(v).strip()}))),
            material_labels=("MATL_CODE_EXP", lambda x: ",".join(sorted({str(v).strip() for v in x.dropna() if str(v).strip()}))),
        )
    )
    agg["total_thickness_mm"] = agg["total_thickness_in"] * INCH_TO_MM
    agg["max_layer_thickness_mm"] = agg["max_layer_thickness_in"] * INCH_TO_MM
    agg["surface_thickness_mm"] = agg["surface_thickness_in"] * INCH_TO_MM
    agg["binder_thickness_mm"] = agg["binder_thickness_in"] * INCH_TO_MM
    agg["base_thickness_mm"] = agg["base_thickness_in"] * INCH_TO_MM
    agg["subbase_thickness_mm"] = agg["subbase_thickness_in"] * INCH_TO_MM
    agg["subgrade_thickness_mm"] = agg["subgrade_thickness_in"] * INCH_TO_MM

    material_text = initial.groupby("node_id_join")["material_text"].apply(lambda s: " || ".join(sorted({str(v) for v in s.dropna() if str(v).strip()}))).reset_index()
    has_ac = initial.groupby("node_id_join")["material_text"].apply(has_ac_material).reset_index(name="has_ac_layer")
    has_pcc = initial.groupby("node_id_join")["material_text"].apply(has_pcc_material).reset_index(name="has_pcc_layer")
    role_flags = initial.groupby("node_id_join").apply(
        lambda group: pd.Series(
            {
                "surface_is_ac": any_role_material(group, "surface", "ac"),
                "surface_is_pcc": any_role_material(group, "surface", "pcc"),
                "has_bound_base": int(
                    (
                        (group["layer_role"] == "base")
                        & group["LAYER_TYPE_EXP"].fillna("").astype(str).str.contains("Bound", case=False)
                    ).any()
                ),
                "has_granular_base": int(
                    (
                        (group["layer_role"] == "base")
                        & group["LAYER_TYPE_EXP"].fillna("").astype(str).str.contains("granular", case=False)
                    ).any()
                ),
                "has_subbase_layer": int((group["layer_role"] == "subbase").any()),
                "has_engineering_fabric": int((group["material_family"] == "fabric").any()),
                "has_stabilized_layer": int(
                    group["LAYER_TYPE_EXP"].fillna("").astype(str).str.contains("treated|bound", case=False).any()
                ),
                "n_bound_layers": int(group["material_family"].isin(["ac", "pcc", "treated"]).sum()),
                "n_unbound_layers": int(group["material_family"].isin(["granular", "soil"]).sum()),
            }
        )
    ).reset_index()

    out = agg.merge(material_text, on="node_id_join", how="left")
    out = out.merge(has_ac, on="node_id_join", how="left")
    out = out.merge(has_pcc, on="node_id_join", how="left")
    out = out.merge(role_flags, on="node_id_join", how="left")
    out["has_ac_layer"] = out["has_ac_layer"].fillna(0).astype(int)
    out["has_pcc_layer"] = out["has_pcc_layer"].fillna(0).astype(int)
    for col in [
        "surface_is_ac",
        "surface_is_pcc",
        "has_bound_base",
        "has_granular_base",
        "has_subbase_layer",
        "has_engineering_fabric",
        "has_stabilized_layer",
        "n_bound_layers",
        "n_unbound_layers",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Loaded materials for {len(out)} sections (first-construction snapshot, anti-leakage)")
    print(f"Wrote {OUT}")
    return out


if __name__ == "__main__":
    load_materials()
