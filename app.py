"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║         SIOP DIGITAL TWIN — Enterprise Knowledge Graph Engine                  ║
║         Tier 1 Automotive Manufacturing | Financial-First Planning             ║
║         Built with Streamlit · Pandas · Plotly                                ║
╚══════════════════════════════════════════════════════════════════════════════════╝

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

# ── Standard library ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Dict
from datetime import datetime, date
import random

# ── Third-party ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ════════════════════════════════════════════════════════════════════════════════
# 0. PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SIOP Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATASET GENERATOR — "ERP Master Data Lake"
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading ERP master data…")
def generate_synthetic_erp_data() -> Dict[str, pd.DataFrame]:
    """
    Generate a realistic Tier-1 automotive manufacturer dataset.

    Returns a dict of DataFrames mirroring common ERP entities:
        - demand_plan   : OEM rolling 12-month demand forecast (units)
        - bom           : Bill of Materials (material cost per unit)
        - inventory     : Current on-hand & on-order balances
        - capacity      : Plant-level monthly capacity (hours available)
        - cost_std      : Standard cost card per SKU
        - freight       : Lane-level freight rates (ocean / air / road)
    """
    np.random.seed(42)
    random.seed(42)

    months = pd.date_range(start="2025-01-01", periods=12, freq="MS")
    month_labels = [m.strftime("%b-%y") for m in months]

    # ── SKU master ───────────────────────────────────────────────────────────────
    skus = {
        "SKU-EV-DRIVE-001": {"name": "EV Drive Module",         "family": "Powertrain",  "asm_hrs": 0.45},
        "SKU-EV-BATT-002":  {"name": "Battery Pack Assembly",   "family": "Powertrain",  "asm_hrs": 0.68},
        "SKU-BRKE-MOD-003": {"name": "Brake Control Module",    "family": "Safety",      "asm_hrs": 0.22},
        "SKU-ADSS-CAM-004": {"name": "ADAS Camera Unit",        "family": "Electronics", "asm_hrs": 0.15},
        "SKU-STEER-005":    {"name": "Electric Power Steering", "family": "Chassis",     "asm_hrs": 0.35},
    }

    # ── OEM demand plan ─────────────────────────────────────────────────────────
    demand_rows = []
    base_demand = {
        "SKU-EV-DRIVE-001": 1800,
        "SKU-EV-BATT-002":  1200,
        "SKU-BRKE-MOD-003": 3500,
        "SKU-ADSS-CAM-004": 4200,
        "SKU-STEER-005":    2800,
    }
    seasonal_idx = np.array([0.88, 0.92, 1.02, 1.10, 1.12, 1.08, 0.95, 0.90, 1.05, 1.10, 1.06, 0.92])

    for sku, base in base_demand.items():
        for i, m in enumerate(months):
            noise = np.random.normal(1.0, 0.04)
            demand_rows.append({
                "month": m,
                "month_label": month_labels[i],
                "sku": sku,
                "sku_name": skus[sku]["name"],
                "family": skus[sku]["family"],
                "demand_units": max(0, int(base * seasonal_idx[i] * noise)),
                "asm_hrs_per_unit": skus[sku]["asm_hrs"],
            })
    demand_df = pd.DataFrame(demand_rows)

    # ── Bill of Materials ───────────────────────────────────────────────────────
    bom_data = {
        "SKU-EV-DRIVE-001": [
            ("Rare Earth Magnets",      "RM-MAG",  85.00, "China",     "Critical"),
            ("Silicon Carbide Chips",   "RM-SIC", 120.00, "Taiwan",    "Critical"),
            ("Copper Winding Assembly", "RM-COP",  42.00, "Mexico",    "Standard"),
            ("Aluminum Housing",        "RM-ALU",  28.00, "Domestic",  "Standard"),
        ],
        "SKU-EV-BATT-002": [
            ("Lithium Cells (NMC)",     "RM-LIT", 310.00, "South Korea","Critical"),
            ("BMS PCB Assembly",        "RM-BMS",  95.00, "Taiwan",     "Critical"),
            ("Thermal Management Sys",  "RM-TMS",  68.00, "Domestic",   "Standard"),
            ("Battery Enclosure",       "RM-ENC",  45.00, "Mexico",     "Standard"),
        ],
        "SKU-BRKE-MOD-003": [
            ("Microcontroller Unit",    "RM-MCU",  22.00, "Taiwan",     "Critical"),
            ("Hydraulic Actuator",      "RM-HYD",  38.00, "Germany",    "Standard"),
            ("Sensor Array",            "RM-SEN",  15.00, "Domestic",   "Standard"),
        ],
        "SKU-ADSS-CAM-004": [
            ("CMOS Image Sensor",       "RM-CMS",  48.00, "Japan",      "Critical"),
            ("GPU Processing Unit",     "RM-GPU",  65.00, "Taiwan",     "Critical"),
            ("Optical Lens Assembly",   "RM-OPT",  25.00, "Germany",    "Standard"),
        ],
        "SKU-STEER-005": [
            ("BLDC Motor Assembly",     "RM-BLD",  72.00, "China",      "Standard"),
            ("Torque Sensor",           "RM-TRQ",  35.00, "Germany",    "Critical"),
            ("ECU Module",              "RM-ECU",  55.00, "Domestic",   "Standard"),
        ],
    }

    bom_rows = []
    for sku, comps in bom_data.items():
        for comp_name, comp_id, cost, origin, risk in comps:
            bom_rows.append({
                "sku": sku,
                "component_name": comp_name,
                "component_id": comp_id,
                "std_cost_usd": cost,
                "origin": origin,
                "supply_risk": risk,
            })
    bom_df = pd.DataFrame(bom_rows)

    # ── Standard cost card ───────────────────────────────────────────────────────
    asm_labor_rate = 55.0
    overhead_rate = 0.28

    cost_rows = []
    for sku, info in skus.items():
        raw_mat = bom_df.loc[bom_df.sku == sku, "std_cost_usd"].sum()
        labor = info["asm_hrs"] * asm_labor_rate
        overhead = (raw_mat + labor) * overhead_rate
        total_cog = raw_mat + labor + overhead
        asp = total_cog * np.random.uniform(1.28, 1.42)
        cost_rows.append({
            "sku": sku,
            "sku_name": info["name"],
            "raw_material_cost": round(raw_mat, 2),
            "direct_labor_cost": round(labor, 2),
            "overhead_cost": round(overhead, 2),
            "total_cogs": round(total_cog, 2),
            "avg_selling_price": round(asp, 2),
            "gross_margin_pct": round((asp - total_cog) / asp * 100, 2),
        })
    cost_df = pd.DataFrame(cost_rows)

    # ── Inventory snapshot ───────────────────────────────────────────────────────
    inv_rows = []
    for sku in skus:
        monthly_demand = demand_df.loc[demand_df.sku == sku, "demand_units"].values
        on_hand = int(monthly_demand[0] * np.random.uniform(1.2, 2.0))
        for i, m in enumerate(months):
            production = int(monthly_demand[i] * np.random.uniform(0.95, 1.08))
            consumption = monthly_demand[i]
            closing_inv = max(0, on_hand + production - consumption)
            inv_rows.append({
                "month": m,
                "month_label": month_labels[i],
                "sku": sku,
                "opening_inv": on_hand,
                "production": production,
                "demand": consumption,
                "closing_inv": closing_inv,
                "days_on_hand": round(closing_inv / max(consumption / 30, 1), 1),
            })
            on_hand = closing_inv
    inv_df = pd.DataFrame(inv_rows)

    # ── Plant capacity ──────────────────────────────────────────────────────────
    plant_hours_per_line = 520
    n_lines = 8
    cap_rows = []
    for i, m in enumerate(months):
        cap_rows.append({
            "month": m,
            "month_label": month_labels[i],
            "available_hours": plant_hours_per_line * n_lines,
            "planned_maintenance_hrs": int(np.random.uniform(40, 90)),
            "n_lines": n_lines,
        })
    cap_df = pd.DataFrame(cap_rows)
    cap_df["net_available_hours"] = cap_df["available_hours"] - cap_df["planned_maintenance_hrs"]

    # ── Freight rates ───────────────────────────────────────────────────────────
    freight_df = pd.DataFrame([
        {"lane": "Taiwan → Plant",      "mode": "Ocean",    "rate_per_cbm": 180,   "lead_time_days": 28},
        {"lane": "Taiwan → Plant",      "mode": "Air",      "rate_per_cbm": 1850,  "lead_time_days": 4},
        {"lane": "South Korea → Plant", "mode": "Ocean",    "rate_per_cbm": 160,   "lead_time_days": 21},
        {"lane": "South Korea → Plant", "mode": "Air",      "rate_per_cbm": 1700,  "lead_time_days": 3},
        {"lane": "Germany → Plant",     "mode": "Road+Sea", "rate_per_cbm": 220,   "lead_time_days": 18},
        {"lane": "China → Plant",       "mode": "Ocean",    "rate_per_cbm": 195,   "lead_time_days": 32},
        {"lane": "China → Plant",       "mode": "Air",      "rate_per_cbm": 2050,  "lead_time_days": 5},
        {"lane": "Mexico → Plant",      "mode": "Road",     "rate_per_cbm": 95,    "lead_time_days": 3},
        {"lane": "Domestic",            "mode": "Road",     "rate_per_cbm": 45,    "lead_time_days": 1},
    ])

    return {
        "demand": demand_df,
        "bom": bom_df,
        "cost": cost_df,
        "inventory": inv_df,
        "capacity": cap_df,
        "freight": freight_df,
    }


# ════════════════════════════════════════════════════════════════════════════════
# 2. DIGITAL BRAIN — Enterprise Knowledge Graph Calculation Engine
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ScenarioParams:
    """Immutable snapshot of all scenario levers from the sidebar."""
    demand_shift_pct: float = 0.0
    rm_cost_spike_pct: float = 0.0
    critical_rm_spike_pct: float = 0.0
    expedite_freight_pct: float = 0.0
    capacity_add_hrs: float = 0.0
    asp_change_pct: float = 0.0
    scrap_rate_delta: float = 0.0
    margin_alert_threshold: float = 20.0
    capacity_alert_pct: float = 95.0
    inv_target_days: float = 30.0


class DigitalTwinEngine:
    def __init__(self, erp_data: Dict[str, pd.DataFrame], params: ScenarioParams):
        self.data = erp_data
        self.params = params

    def build_scenario_demand(self) -> pd.DataFrame:
        df = self.data["demand"].copy()
        shift = 1.0 + self.params.demand_shift_pct / 100.0
        df["scenario_demand_units"] = (df["demand_units"] * shift).round(0).astype(int)
        df["demand_delta_units"] = df["scenario_demand_units"] - df["demand_units"]
        return df

    def build_scenario_costs(self) -> pd.DataFrame:
        bom = self.data["bom"].copy()
        cost = self.data["cost"].copy()
        p = self.params

        broad_mult = 1.0 + p.rm_cost_spike_pct / 100.0
        bom["scenario_rm_cost"] = bom["std_cost_usd"] * broad_mult

        crit_mult = 1.0 + p.critical_rm_spike_pct / 100.0
        mask = bom["supply_risk"] == "Critical"
        bom.loc[mask, "scenario_rm_cost"] *= crit_mult

        rm_by_sku = (
            bom.groupby("sku", as_index=False)["scenario_rm_cost"]
               .sum()
               .rename(columns={"scenario_rm_cost": "scenario_rm_total"})
        )
        sc = cost.merge(rm_by_sku, on="sku", how="left")

        air_premium_per_unit = 18.0
        sc["scenario_freight_adder"] = air_premium_per_unit * (p.expedite_freight_pct / 100.0)

        sc["scenario_scrap_adder"] = sc["scenario_rm_total"] * (p.scrap_rate_delta / 100.0)

        rm_delta = sc["scenario_rm_total"] - sc["raw_material_cost"]
        sc["scenario_cogs"] = (
            sc["total_cogs"]
            + rm_delta
            + sc["scenario_freight_adder"]
            + sc["scenario_scrap_adder"]
        ).round(2)

        asp_mult = 1.0 + p.asp_change_pct / 100.0
        sc["scenario_asp"] = (sc["avg_selling_price"] * asp_mult).round(2)

        sc["scenario_gm_pct"] = (
            (sc["scenario_asp"] - sc["scenario_cogs"]) / sc["scenario_asp"] * 100
        ).round(2)

        sc["baseline_gm_pct"] = sc["gross_margin_pct"]
        sc["gm_delta_ppts"] = (sc["scenario_gm_pct"] - sc["baseline_gm_pct"]).round(2)

        return sc

    def build_capacity_analysis(self, scenario_demand: pd.DataFrame) -> pd.DataFrame:
        cap = self.data["capacity"].copy()
        p = self.params

        # Vectorized required hours
        req = scenario_demand.copy()
        req["req_hrs"] = req["scenario_demand_units"] * req["asm_hrs_per_unit"]
        required_hrs = req.groupby("month", as_index=False)["req_hrs"].sum().rename(columns={"req_hrs": "required_asm_hrs"})

        cap = cap.merge(required_hrs, on="month", how="left")
        cap["required_asm_hrs"] = cap["required_asm_hrs"].fillna(0.0)

        cap["ot_hours_available"] = float(p.capacity_add_hrs)
        cap["total_available_hrs"] = cap["net_available_hours"] + cap["ot_hours_available"]

        cap["utilisation_pct"] = (cap["required_asm_hrs"] / cap["total_available_hrs"] * 100).round(2)

        cap["capacity_gap_hrs"] = (cap["required_asm_hrs"] - cap["total_available_hrs"]).round(1)
        cap["capacity_status"] = cap["capacity_gap_hrs"].apply(lambda x: "🔴 CONSTRAINED" if x > 0 else "🟢 FEASIBLE")
        cap["utilisation_alert"] = cap["utilisation_pct"] >= p.capacity_alert_pct

        return cap

    def build_pl_bridge(self, scenario_demand: pd.DataFrame, scenario_costs: pd.DataFrame) -> Dict[str, float]:
        merged = scenario_demand.merge(
            scenario_costs[["sku", "total_cogs", "avg_selling_price", "scenario_cogs", "scenario_asp"]],
            on="sku",
            how="left"
        )

        merged["baseline_rev"] = merged["demand_units"] * merged["avg_selling_price"]
        merged["baseline_cogs"] = merged["demand_units"] * merged["total_cogs"]

        merged["scenario_rev"] = merged["scenario_demand_units"] * merged["scenario_asp"]
        merged["scenario_cogs_total"] = merged["scenario_demand_units"] * merged["scenario_cogs"]

        baseline_rev = merged["baseline_rev"].sum()
        baseline_cogs = merged["baseline_cogs"].sum()
        baseline_gp = baseline_rev - baseline_cogs

        scenario_rev = merged["scenario_rev"].sum()
        scenario_cogs = merged["scenario_cogs_total"].sum()
        scenario_gp = scenario_rev - scenario_cogs

        # Bridge components
        volume_rev_impact = (merged["demand_delta_units"] * merged["avg_selling_price"]).sum()
        price_impact = (merged["scenario_demand_units"] * (merged["scenario_asp"] - merged["avg_selling_price"])).sum()
        rm_cost_impact = -(merged["scenario_demand_units"] * (merged["scenario_cogs"] - merged["total_cogs"])).sum()

        return {
            "baseline_rev": round(baseline_rev, 2),
            "baseline_cogs": round(baseline_cogs, 2),
            "baseline_gp": round(baseline_gp, 2),
            "baseline_gm_pct": round(baseline_gp / baseline_rev * 100, 2) if baseline_rev else 0.0,
            "scenario_rev": round(scenario_rev, 2),
            "scenario_cogs": round(scenario_cogs, 2),
            "scenario_gp": round(scenario_gp, 2),
            "scenario_gm_pct": round(scenario_gp / scenario_rev * 100, 2) if scenario_rev else 0.0,
            "volume_rev_impact": round(volume_rev_impact, 2),
            "price_impact": round(price_impact, 2),
            "rm_cost_impact": round(rm_cost_impact, 2),
            "gp_delta": round(scenario_gp - baseline_gp, 2),
        }

    def build_inventory_projection(self, scenario_demand: pd.DataFrame, scenario_costs: pd.DataFrame) -> pd.DataFrame:
        inv = self.data["inventory"].copy()

        sc_inv = inv.merge(
            scenario_demand[["month", "sku", "scenario_demand_units"]],
            on=["month", "sku"],
            how="left"
        )
        sc_inv = sc_inv.sort_values(["sku", "month"]).reset_index(drop=True)

        rows = []
        for sku in sc_inv["sku"].unique():
            sub = sc_inv.loc[sc_inv.sku == sku].copy()
            oh = float(sub.iloc[0]["opening_inv"])
            for _, row in sub.iterrows():
                prod = float(row["production"])
                sc_dem = float(row["scenario_demand_units"])
                close = max(0.0, oh + prod - sc_dem)
                out = row.to_dict()
                out.update({
                    "sc_opening_inv": oh,
                    "sc_closing_inv": close,
                    "sc_days_on_hand": round(close / max(sc_dem / 30.0, 1.0), 1),
                })
                rows.append(out)
                oh = close

        sc_inv_df = pd.DataFrame(rows)

        baseline_cost_map = self.data["cost"].set_index("sku")["total_cogs"].to_dict()
        scenario_cost_map = scenario_costs.set_index("sku")["scenario_cogs"].to_dict()

        sc_inv_df["baseline_inv_value"] = sc_inv_df["closing_inv"] * sc_inv_df["sku"].map(baseline_cost_map)
        sc_inv_df["scenario_inv_value"] = sc_inv_df["sc_closing_inv"] * sc_inv_df["sku"].map(scenario_cost_map)

        return sc_inv_df

    def detect_exceptions(self, scenario_costs: pd.DataFrame, capacity_df: pd.DataFrame, pl_bridge: Dict[str, float]) -> pd.DataFrame:
        p = self.params
