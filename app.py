"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║         SIOP DIGITAL TWIN — SIOP DIGITAL TWIN                  ║
║         Tier 1 Automotive Manufacturing | DEMAND SUPPLY INVENTORY SIOP             ║
║         Built with Streamlit · Pandas · Plotly                                ║
╚══════════════════════════════════════════════════════════════════════════════════╝

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

# ── Standard library ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Dict, Tuple
import random

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
#    Mimics data extracted from SAP S/4HANA / Oracle Fusion
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
    # Assembly hours per unit: realistic Tier-1 automotive line-rate values
    # (includes takt time, test cycle, material handling)
    skus = {
        "SKU-EV-DRIVE-001": {"name": "Bumeper",        "family": "Powertrain",  "asm_hrs": 0.45},
        "SKU-EV-BATT-002":  {"name": "Battery Pack Assembly",  "family": "Powertrain",  "asm_hrs": 0.68},
        "SKU-BRKE-MOD-003": {"name": "Brake Control Module",   "family": "Safety",      "asm_hrs": 0.22},
        "SKU-ADSS-CAM-004": {"name": "ADAS Camera Unit",       "family": "Electronics", "asm_hrs": 0.15},
        "SKU-STEER-005":    {"name": "Electric Power Steering", "family": "Chassis",    "asm_hrs": 0.35},
    }

    # ── OEM demand plan (units/month per SKU) ────────────────────────────────────
    demand_rows = []
    base_demand = {
        "SKU-EV-DRIVE-001": 1_800,
        "SKU-EV-BATT-002":  1_200,
        "SKU-BRKE-MOD-003": 3_500,
        "SKU-ADSS-CAM-004": 4_200,
        "SKU-STEER-005":    2_800,
    }
    seasonal_idx = np.array([0.88, 0.92, 1.02, 1.10, 1.12, 1.08,
                              0.95, 0.90, 1.05, 1.10, 1.06, 0.92])

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

    # ── Bill of Materials — raw material cost per unit (USD) ─────────────────────
    bom_data = {
        "SKU-EV-DRIVE-001": [
            ("Rare Earth Magnets",      "RM-MAG", 85.00, "China",       "Critical"),
            ("Silicon Carbide Chips",   "RM-SIC", 120.00,"Taiwan",      "Critical"),
            ("Copper Winding Assembly", "RM-COP", 42.00, "Mexico",      "Standard"),
            ("Aluminum Housing",        "RM-ALU", 28.00, "Domestic",    "Standard"),
        ],
        "SKU-EV-BATT-002": [
            ("Lithium Cells (NMC)",     "RM-LIT", 310.00,"South Korea", "Critical"),
            ("BMS PCB Assembly",        "RM-BMS", 95.00, "Taiwan",      "Critical"),
            ("Thermal Management Sys",  "RM-TMS", 68.00, "Domestic",    "Standard"),
            ("Battery Enclosure",       "RM-ENC", 45.00, "Mexico",      "Standard"),
        ],
        "SKU-BRKE-MOD-003": [
            ("Microcontroller Unit",    "RM-MCU", 22.00, "Taiwan",      "Critical"),
            ("Hydraulic Actuator",      "RM-HYD", 38.00, "Germany",     "Standard"),
            ("Sensor Array",            "RM-SEN", 15.00, "Domestic",    "Standard"),
        ],
        "SKU-ADSS-CAM-004": [
            ("CMOS Image Sensor",       "RM-CMS", 48.00, "Japan",       "Critical"),
            ("GPU Processing Unit",     "RM-GPU", 65.00, "Taiwan",      "Critical"),
            ("Optical Lens Assembly",   "RM-OPT", 25.00, "Germany",     "Standard"),
        ],
        "SKU-STEER-005": [
            ("BLDC Motor Assembly",     "RM-BLD", 72.00, "China",       "Standard"),
            ("Torque Sensor",           "RM-TRQ", 35.00, "Germany",     "Critical"),
            ("ECU Module",              "RM-ECU", 55.00, "Domestic",    "Standard"),
        ],
    }
    bom_rows = []
    for sku, comps in bom_data.items():
        for comp_name, comp_id, cost, origin, risk in comps:
            bom_rows.append({
                "sku": sku, "component_name": comp_name, "component_id": comp_id,
                "std_cost_usd": cost, "origin": origin, "supply_risk": risk,
            })
    bom_df = pd.DataFrame(bom_rows)

    # ── Standard cost card ───────────────────────────────────────────────────────
    asm_labor_rate = 55.0   # USD / assembly hour
    overhead_rate  = 0.28   # 28 % of direct cost

    cost_rows = []
    for sku, info in skus.items():
        raw_mat  = bom_df[bom_df.sku == sku]["std_cost_usd"].sum()
        labor    = info["asm_hrs"] * asm_labor_rate
        overhead = (raw_mat + labor) * overhead_rate
        total_cog= raw_mat + labor + overhead
        asp      = total_cog * np.random.uniform(1.28, 1.42)
        cost_rows.append({
            "sku": sku, "sku_name": info["name"],
            "raw_material_cost": round(raw_mat, 2),
            "direct_labor_cost": round(labor, 2),
            "overhead_cost":     round(overhead, 2),
            "total_cogs":        round(total_cog, 2),
            "avg_selling_price": round(asp, 2),
            "gross_margin_pct":  round((asp - total_cog) / asp * 100, 2),
        })
    cost_df = pd.DataFrame(cost_rows)

    # ── Inventory snapshot ───────────────────────────────────────────────────────
    inv_rows = []
    for sku in skus:
        monthly_demand = demand_df[demand_df.sku == sku]["demand_units"].values
        on_hand = int(monthly_demand[0] * np.random.uniform(1.2, 2.0))
        for i, m in enumerate(months):
            production  = int(monthly_demand[i] * np.random.uniform(0.95, 1.08))
            consumption = monthly_demand[i]
            closing_inv = max(0, on_hand + production - consumption)
            inv_rows.append({
                "month": m, "month_label": month_labels[i], "sku": sku,
                "opening_inv": on_hand, "production": production,
                "demand": consumption, "closing_inv": closing_inv,
                "days_on_hand": round(closing_inv / max(consumption / 30, 1), 1),
            })
            on_hand = closing_inv
    inv_df = pd.DataFrame(inv_rows)

    # ── Plant capacity (8 dedicated assembly lines) ──────────────────────────────
    # 520h ≈ 22 working days × 2 shifts × 11.8 h effective (OEE-adjusted)
    plant_hours_per_line = 520
    n_lines = 8
    cap_rows = []
    for i, m in enumerate(months):
        cap_rows.append({
            "month": m, "month_label": month_labels[i],
            "available_hours": plant_hours_per_line * n_lines,
            "planned_maintenance_hrs": int(np.random.uniform(40, 90)),
            "n_lines": n_lines,
        })
    cap_df = pd.DataFrame(cap_rows)
    cap_df["net_available_hours"] = (
        cap_df["available_hours"] - cap_df["planned_maintenance_hrs"]
    )

    # ── Freight rates (baseline) ─────────────────────────────────────────────────
    freight_df = pd.DataFrame([
        {"lane": "Taiwan → Plant",     "mode": "Ocean",    "rate_per_cbm": 180,   "lead_time_days": 28},
        {"lane": "Taiwan → Plant",     "mode": "Air",      "rate_per_cbm": 1_850, "lead_time_days": 4},
        {"lane": "South Korea → Plant","mode": "Ocean",    "rate_per_cbm": 160,   "lead_time_days": 21},
        {"lane": "South Korea → Plant","mode": "Air",      "rate_per_cbm": 1_700, "lead_time_days": 3},
        {"lane": "Germany → Plant",    "mode": "Road+Sea", "rate_per_cbm": 220,   "lead_time_days": 18},
        {"lane": "China → Plant",      "mode": "Ocean",    "rate_per_cbm": 195,   "lead_time_days": 32},
        {"lane": "China → Plant",      "mode": "Air",      "rate_per_cbm": 2_050, "lead_time_days": 5},
        {"lane": "Mexico → Plant",     "mode": "Road",     "rate_per_cbm": 95,    "lead_time_days": 3},
        {"lane": "Domestic",           "mode": "Road",     "rate_per_cbm": 45,    "lead_time_days": 1},
    ])

    return {
        "demand":   demand_df,
        "bom":      bom_df,
        "cost":     cost_df,
        "inventory": inv_df,
        "capacity": cap_df,
        "freight":  freight_df,
    }


# ════════════════════════════════════════════════════════════════════════════════
# 2. DIGITAL BRAIN — Enterprise Knowledge Graph Calculation Engine
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioParams:
    """Immutable snapshot of all scenario levers from the sidebar."""
    demand_shift_pct:       float = 0.0
    rm_cost_spike_pct:      float = 0.0
    critical_rm_spike_pct:  float = 0.0
    expedite_freight_pct:   float = 0.0
    capacity_add_hrs:       float = 0.0
    asp_change_pct:         float = 0.0
    scrap_rate_delta:       float = 0.0
    margin_alert_threshold: float = 20.0
    capacity_alert_pct:     float = 95.0
    inv_target_days:        float = 30.0


class DigitalTwinEngine:
    """
    Core vectorized calculation engine.
    Links operational decisions (volume, sourcing, capacity) to P&L / balance sheet.
    All computation uses Pandas/NumPy — no Python row-level loops.
    """

    def __init__(self, erp_data: Dict[str, pd.DataFrame], params: ScenarioParams):
        self.data   = erp_data
        self.params = params

    # ── 2a. Demand Plan ──────────────────────────────────────────────────────────
    def build_scenario_demand(self) -> pd.DataFrame:
        df = self.data["demand"].copy()
        shift = 1.0 + self.params.demand_shift_pct / 100.0
        df["scenario_demand_units"] = (df["demand_units"] * shift).round(0).astype(int)
        df["demand_delta_units"]    = df["scenario_demand_units"] - df["demand_units"]
        return df

    # ── 2b. Material Cost Recalculation ─────────────────────────────────────────
    def build_scenario_costs(self) -> pd.DataFrame:
        bom  = self.data["bom"].copy()
        cost = self.data["cost"].copy()
        p    = self.params

        # Broad RM spike (all components)
        broad_mult = 1.0 + p.rm_cost_spike_pct / 100.0
        bom["scenario_rm_cost"] = bom["std_cost_usd"] * broad_mult

        # Critical component overlay (additive on top of broad)
        crit_mult = 1.0 + p.critical_rm_spike_pct / 100.0
        mask = bom["supply_risk"] == "Critical"
        bom.loc[mask, "scenario_rm_cost"] *= crit_mult

        # Roll up to SKU level
        rm_by_sku = (
            bom.groupby("sku")["scenario_rm_cost"]
               .sum()
               .reset_index()
               .rename(columns={"scenario_rm_cost": "scenario_rm_total"})
        )

        sc = cost.merge(rm_by_sku, on="sku", how="left")

        # Air freight premium (avg $18 / unit on expedited lanes)
        air_premium_per_unit = 18.0
        sc["scenario_freight_adder"] = air_premium_per_unit * (p.expedite_freight_pct / 100.0)

        # Scrap yield loss
        sc["scenario_scrap_adder"] = sc["scenario_rm_total"] * (p.scrap_rate_delta / 100.0)

        # Rebuild COGS
        rm_delta = sc["scenario_rm_total"] - sc["raw_material_cost"]
        sc["scenario_cogs"] = (
            sc["total_cogs"]
            + rm_delta
            + sc["scenario_freight_adder"]
            + sc["scenario_scrap_adder"]
        ).round(2)

        # ASP change (pass-through pricing)
        asp_mult = 1.0 + p.asp_change_pct / 100.0
        sc["scenario_asp"] = (sc["avg_selling_price"] * asp_mult).round(2)

        sc["scenario_gm_pct"] = (
            (sc["scenario_asp"] - sc["scenario_cogs"]) / sc["scenario_asp"] * 100
        ).round(2)

        sc["baseline_gm_pct"] = sc["gross_margin_pct"]
        sc["gm_delta_ppts"]   = (sc["scenario_gm_pct"] - sc["baseline_gm_pct"]).round(2)

        return sc

    # ── 2c. Capacity & Utilisation ───────────────────────────────────────────────
    def build_capacity_analysis(self, scenario_demand: pd.DataFrame) -> pd.DataFrame:
        cap = self.data["capacity"].copy()
        p   = self.params

        required_hrs = (
            scenario_demand
            .groupby("month")
            .apply(lambda g: (g["scenario_demand_units"] * g["asm_hrs_per_unit"]).sum())
            .reset_index()
            .rename(columns={0: "required_asm_hrs"})
        )

        cap = cap.merge(required_hrs, on="month", how="left")
        cap["ot_hours_available"]  = p.capacity_add_hrs
        cap["total_available_hrs"] = cap["net_available_hours"] + cap["ot_hours_available"]

        cap["utilisation_pct"] = (
            cap["required_asm_hrs"] / cap["total_available_hrs"] * 100
        ).round(2)

        cap["capacity_gap_hrs"] = (
            cap["required_asm_hrs"] - cap["total_available_hrs"]
        ).round(1)

        cap["capacity_status"] = cap["capacity_gap_hrs"].apply(
            lambda x: "🔴 CONSTRAINED" if x > 0 else "🟢 FEASIBLE"
        )
        cap["utilisation_alert"] = cap["utilisation_pct"] >= p.capacity_alert_pct
        return cap

    # ── 2d. P&L Bridge ───────────────────────────────────────────────────────────
    def build_pl_bridge(
        self,
        scenario_demand: pd.DataFrame,
        scenario_costs:  pd.DataFrame,
    ) -> Dict[str, float]:
        merged = scenario_demand.merge(
            scenario_costs[["sku", "total_cogs", "avg_selling_price",
                             "scenario_cogs", "scenario_asp"]],
            on="sku"
        )

        merged["baseline_rev"]        = merged["demand_units"]          * merged["avg_selling_price"]
        merged["baseline_cogs_total"] = merged["demand_units"]          * merged["total_cogs"]
        merged["scenario_rev"]        = merged["scenario_demand_units"] * merged["scenario_asp"]
        merged["scenario_cogs_total"] = merged["scenario_demand_units"] * merged["scenario_cogs"]

        baseline_rev  = merged["baseline_rev"].sum()
        baseline_cogs = merged["baseline_cogs_total"].sum()
        baseline_gp   = baseline_rev - baseline_cogs

        scenario_rev  = merged["scenario_rev"].sum()
        scenario_cogs = merged["scenario_cogs_total"].sum()
        scenario_gp   = scenario_rev - scenario_cogs

        volume_rev_impact = (
            (merged["demand_units"] * self.params.demand_shift_pct / 100)
            * merged["avg_selling_price"]
        ).sum()

        price_impact = (
            merged["scenario_demand_units"]
            * (merged["scenario_asp"] - merged["avg_selling_price"])
        ).sum()

        rm_cost_impact = -(
            merged["scenario_demand_units"]
            * (merged["scenario_cogs"] - merged["total_cogs"])
        ).sum()

        return {
            "baseline_rev":      round(baseline_rev,  2),
            "baseline_cogs":     round(baseline_cogs, 2),
            "baseline_gp":       round(baseline_gp,   2),
            "baseline_gm_pct":   round(baseline_gp / max(baseline_rev, 1) * 100, 2),
            "scenario_rev":      round(scenario_rev,  2),
            "scenario_cogs":     round(scenario_cogs, 2),
            "scenario_gp":       round(scenario_gp,   2),
            "scenario_gm_pct":   round(scenario_gp / max(scenario_rev, 1) * 100, 2),
            "volume_rev_impact": round(volume_rev_impact, 2),
            "price_impact":      round(price_impact,  2),
            "rm_cost_impact":    round(rm_cost_impact, 2),
            "gp_delta":          round(scenario_gp - baseline_gp, 2),
        }

    # ── 2e. Inventory Projection & Working Capital ───────────────────────────────
    def build_inventory_projection(self, scenario_demand: pd.DataFrame) -> pd.DataFrame:
        inv = self.data["inventory"].copy()

        sc_inv = inv.merge(
            scenario_demand[["month", "sku", "scenario_demand_units", "demand_units"]],
            on=["month", "sku"], how="left"
        ).sort_values(["sku", "month"]).reset_index(drop=True)

        rows = []
        for sku in sc_inv["sku"].unique():
            sub = sc_inv[sc_inv.sku == sku].copy()
            oh  = int(sub.iloc[0]["opening_inv"])
            for _, row in sub.iterrows():
                prod   = int(row["production"])
                sc_dem = int(row["scenario_demand_units"])
                close  = max(0, oh + prod - sc_dem)
                rows.append({
                    **row.to_dict(),
                    "sc_opening_inv": oh,
                    "sc_closing_inv": close,
                    "sc_days_on_hand": round(close / max(sc_dem / 30, 1), 1),
                })
                oh = close

        sc_inv_df = pd.DataFrame(rows)
        cost_map  = self.data["cost"].set_index("sku")["total_cogs"].to_dict()

        sc_inv_df["baseline_inv_value"] = (
            sc_inv_df["closing_inv"]    * sc_inv_df["sku"].map(cost_map)
        )
        sc_inv_df["scenario_inv_value"] = (
            sc_inv_df["sc_closing_inv"] * sc_inv_df["sku"].map(cost_map)
        )
        return sc_inv_df

    # ── 2f. Exception Engine ─────────────────────────────────────────────────────
    def detect_exceptions(
        self,
        scenario_costs: pd.DataFrame,
        capacity_df:    pd.DataFrame,
        pl_bridge:      Dict[str, float],
    ) -> pd.DataFrame:
        p = self.params
        exc_rows = []

        # Per-SKU margin alerts
        for _, row in scenario_costs.iterrows():
            if row["scenario_gm_pct"] < p.margin_alert_threshold:
                exc_rows.append({
                    "severity":  "🔴 CRITICAL",
                    "category":  "Margin",
                    "entity":    row["sku_name"],
                    "metric":    f"Gross Margin: {row['scenario_gm_pct']:.1f}%",
                    "threshold": f"< {p.margin_alert_threshold:.0f}%",
                    "action":    "Review pricing or reduce RM exposure",
                })

        # Per-month capacity alerts
        for _, row in capacity_df[capacity_df["utilisation_alert"]].iterrows():
            exc_rows.append({
                "severity":  "🟠 WARNING",
                "category":  "Capacity",
                "entity":    f"Plant — {row['month_label']}",
                "metric":    f"Utilisation: {row['utilisation_pct']:.1f}%",
                "threshold": f"≥ {p.capacity_alert_pct:.0f}%",
                "action":    "Add OT / defer low-margin volume",
            })

        # Portfolio-level GM degradation
        gm_drop = pl_bridge["baseline_gm_pct"] - pl_bridge["scenario_gm_pct"]
        if gm_drop > 2.0:
            exc_rows.append({
                "severity":  "🟠 WARNING",
                "category":  "P&L",
                "entity":    "Portfolio",
                "metric":    f"GM degradation: -{gm_drop:.1f} ppts",
                "threshold": "> 2 ppts vs baseline",
                "action":    "Escalate to S&OP leadership",
            })

        if not exc_rows:
            exc_rows.append({
                "severity":  "🟢 CLEAR",
                "category":  "All",
                "entity":    "No exceptions",
                "metric":    "Plan within all thresholds",
                "threshold": "N/A",
                "action":    "Proceed to executive review",
            })

        return pd.DataFrame(exc_rows)

    # ── 2g. Full run ─────────────────────────────────────────────────────────────
    def run(self) -> Dict:
        sc_demand  = self.build_scenario_demand()
        sc_costs   = self.build_scenario_costs()
        capacity   = self.build_capacity_analysis(sc_demand)
        pl_bridge  = self.build_pl_bridge(sc_demand, sc_costs)
        inventory  = self.build_inventory_projection(sc_demand)
        exceptions = self.detect_exceptions(sc_costs, capacity, pl_bridge)
        return {
            "sc_demand":  sc_demand,
            "sc_costs":   sc_costs,
            "capacity":   capacity,
            "pl_bridge":  pl_bridge,
            "inventory":  inventory,
            "exceptions": exceptions,
        }


# ════════════════════════════════════════════════════════════════════════════════
# 3. CHART LIBRARY — Plotly Visualizations
#
# FIX: CHART_LAYOUT no longer includes 'xaxis', 'yaxis', or 'margin'.
#      Those keys are applied per-chart via fig.update_xaxes() /
#      fig.update_yaxes() / fig.update_layout(margin=…) to prevent the
#      "duplicate keyword argument" TypeError when a chart needs to override
#      axis properties (e.g. yaxis range in chart_capacity_utilisation).
# ════════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "baseline": "#3B82F6",
    "scenario": "#F59E0B",
    "positive": "#10B981",
    "negative": "#EF4444",
    "neutral":  "#6B7280",
    "bg":       "rgba(0,0,0,0)",
    "grid":     "rgba(255,255,255,0.08)",
    "text":     "#E5E7EB",
}

# Shared layout — intentionally excludes xaxis/yaxis/margin so charts
# can add their own without triggering duplicate-keyword errors.
CHART_LAYOUT_BASE = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor =PALETTE["bg"],
    font=dict(family="Inter, sans-serif", size=12, color=PALETTE["text"]),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", borderwidth=0),
)

# Default margin reused by most charts (applied explicitly, not via **spread)
DEFAULT_MARGIN = dict(l=20, r=20, t=50, b=30)


def _base_layout(title: str, yaxis_title: str = "", extra: dict = None) -> dict:
    """
    Build a complete layout dict without any key conflicts.
    Per-chart overrides go in `extra` (e.g. {'barmode': 'group'}).
    """
    layout = {
        **CHART_LAYOUT_BASE,
        "title":       title,
        "yaxis_title": yaxis_title,
        "margin":      DEFAULT_MARGIN,
    }
    if extra:
        layout.update(extra)
    return layout


def _style_axes(fig: go.Figure, y_range: list = None) -> go.Figure:
    """Apply shared grid styling to all axes after layout is set."""
    fig.update_xaxes(gridcolor=PALETTE["grid"], showgrid=True)
    fig.update_yaxes(gridcolor=PALETTE["grid"], showgrid=True)
    if y_range:
        fig.update_yaxes(range=y_range)
    return fig


# ── Chart 1: Demand Comparison ───────────────────────────────────────────────
def chart_demand_comparison(sc_demand: pd.DataFrame) -> go.Figure:
    """Grouped bar: Baseline vs Scenario demand by month (all SKUs aggregated)."""
    agg = (
        sc_demand
        .groupby("month_label")[["demand_units", "scenario_demand_units"]]
        .sum()
        .reset_index()
    )
    fig = go.Figure([
        go.Bar(
            name="Baseline Demand",
            x=agg["month_label"], y=agg["demand_units"],
            marker_color=PALETTE["baseline"], opacity=0.85,
        ),
        go.Bar(
            name="Scenario Demand",
            x=agg["month_label"], y=agg["scenario_demand_units"],
            marker_color=PALETTE["scenario"], opacity=0.85,
        ),
    ])
    fig.update_layout(**_base_layout(
        "📦 Demand Plan — Baseline vs Scenario (All SKUs)",
        yaxis_title="Units",
        extra={"barmode": "group"},
    ))
    return _style_axes(fig)


# ── Chart 2: Capacity Utilisation ────────────────────────────────────────────
def chart_capacity_utilisation(capacity_df: pd.DataFrame, alert_pct: float) -> go.Figure:
    """Area + line chart: capacity utilisation % with configurable alert band."""
    fig = go.Figure()

    # Alert shading band
    fig.add_hrect(
        y0=alert_pct, y1=115,
        fillcolor="rgba(239,68,68,0.12)", line_width=0,
        annotation_text=f"Alert ≥{alert_pct:.0f}%",
        annotation_position="top right",
        annotation_font=dict(color="#EF4444", size=11),
    )

    point_colors = [
        PALETTE["negative"] if v else PALETTE["positive"]
        for v in capacity_df["utilisation_alert"]
    ]

    fig.add_trace(go.Scatter(
        x=capacity_df["month_label"],
        y=capacity_df["utilisation_pct"],
        mode="lines+markers",
        line=dict(color=PALETTE["scenario"], width=3),
        marker=dict(size=8, color=point_colors),
        name="Utilisation %",
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.12)",
    ))

    fig.add_hline(
        y=100, line_dash="dash", line_color=PALETTE["negative"],
        annotation_text="100% Capacity",
        annotation_position="bottom right",
    )

    # Use update_layout WITHOUT a yaxis key — set range via update_yaxes instead
    fig.update_layout(**_base_layout(
        "🏭 Plant Capacity Utilisation — Scenario Plan",
        yaxis_title="Utilisation %",
    ))

    # Now safely set the y-axis range without any conflict
    max_util = float(capacity_df["utilisation_pct"].max())
    _style_axes(fig, y_range=[0, max(115, max_util + 5)])
    return fig


# ── Chart 3: P&L Waterfall ───────────────────────────────────────────────────
def chart_pl_waterfall(pl_bridge: Dict[str, float]) -> go.Figure:
    """Gross Profit Bridge waterfall: Baseline → Volume → Price → RM Cost → Scenario."""
    M = 1_000_000

    labels  = ["Baseline GP",  "Volume Impact", "Price / ASP",
               "RM & Freight", "Scenario GP"]
    values  = [
        pl_bridge["baseline_gp"]       / M,
        pl_bridge["volume_rev_impact"] / M,
        pl_bridge["price_impact"]      / M,
        pl_bridge["rm_cost_impact"]    / M,
        pl_bridge["scenario_gp"]       / M,
    ]
    measure = ["absolute", "relative", "relative", "relative", "total"]

    fig = go.Figure(go.Waterfall(
        name="P&L Bridge",
        orientation="v",
        measure=measure,
        x=labels,
        y=values,
        connector=dict(line=dict(color="#6B7280", width=1, dash="dot")),
        decreasing=dict(marker_color=PALETTE["negative"]),
        increasing=dict(marker_color=PALETTE["positive"]),
        totals=dict(marker_color=PALETTE["scenario"]),
        text=[f"${v:.2f}M" for v in values],
        textposition="outside",
    ))

    fig.update_layout(**_base_layout(
        "💰 Gross Profit Bridge — Baseline → Scenario (12-Month, $M)",
        yaxis_title="Gross Profit ($M)",
        extra={"showlegend": False},
    ))
    return _style_axes(fig)


# ── Chart 4: Inventory Trend ─────────────────────────────────────────────────
def chart_inventory_trend(inv_df: pd.DataFrame) -> go.Figure:
    """Dual line: Closing inventory units — Baseline vs Scenario (all SKUs)."""
    agg = (
        inv_df
        .groupby("month_label")[["closing_inv", "sc_closing_inv"]]
        .sum()
        .reset_index()
    )
    fig = go.Figure([
        go.Scatter(
            x=agg["month_label"], y=agg["closing_inv"],
            mode="lines+markers", name="Baseline Inventory",
            line=dict(color=PALETTE["baseline"], width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        ),
        go.Scatter(
            x=agg["month_label"], y=agg["sc_closing_inv"],
            mode="lines+markers", name="Scenario Inventory",
            line=dict(color=PALETTE["scenario"], width=2.5),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
        ),
    ])
    fig.update_layout(**_base_layout(
        "📊 Inventory Balance — Closing Stock (All SKUs)",
        yaxis_title="Units",
    ))
    return _style_axes(fig)


# ── Chart 5: Margin by SKU ───────────────────────────────────────────────────
def chart_margin_by_sku(sc_costs: pd.DataFrame, margin_threshold: float) -> go.Figure:
    """Horizontal grouped bar: Baseline vs Scenario GM% per SKU."""
    bar_colors_sc = [
        PALETTE["negative"] if v < margin_threshold else PALETTE["scenario"]
        for v in sc_costs["scenario_gm_pct"]
    ]

    fig = go.Figure([
        go.Bar(
            name="Baseline GM%",
            x=sc_costs["baseline_gm_pct"],
            y=sc_costs["sku_name"],
            orientation="h",
            marker_color=PALETTE["baseline"],
            opacity=0.85,
            text=sc_costs["baseline_gm_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ),
        go.Bar(
            name="Scenario GM%",
            x=sc_costs["scenario_gm_pct"],
            y=sc_costs["sku_name"],
            orientation="h",
            marker_color=bar_colors_sc,
            opacity=0.85,
            text=sc_costs["scenario_gm_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ),
    ])

    # Use a separate margin here — does NOT conflict because CHART_LAYOUT_BASE
    # no longer carries a 'margin' key.
    fig.update_layout(**_base_layout(
        "📈 Gross Margin % by SKU — Baseline vs Scenario",
        yaxis_title="",
        extra={
            "barmode": "group",
            "xaxis_title": "Gross Margin (%)",
            "height": 320,
            "margin": dict(l=20, r=90, t=50, b=20),   # override default margin safely
        },
    ))
    return _style_axes(fig)


# ── Chart 6: Working Capital ─────────────────────────────────────────────────
def chart_working_capital(inv_df: pd.DataFrame) -> go.Figure:
    """Filled area: Inventory working capital ($M) — Baseline vs Scenario."""
    agg = (
        inv_df
        .groupby("month_label")[["baseline_inv_value", "scenario_inv_value"]]
        .sum()
        .reset_index()
    )
    agg[["baseline_inv_value", "scenario_inv_value"]] /= 1_000_000

    fig = go.Figure([
        go.Scatter(
            x=agg["month_label"], y=agg["baseline_inv_value"],
            mode="lines", name="Baseline WC ($M)",
            line=dict(color=PALETTE["baseline"], width=2, dash="dot"),
        ),
        go.Scatter(
            x=agg["month_label"], y=agg["scenario_inv_value"],
            mode="lines", name="Scenario WC ($M)",
            line=dict(color=PALETTE["scenario"], width=2.5),
            fill="tonexty", fillcolor="rgba(245,158,11,0.10)",
        ),
    ])
    fig.update_layout(**_base_layout(
        "💳 Working Capital in Inventory ($M) — 12-Month Horizon",
        yaxis_title="$ Million",
    ))
    return _style_axes(fig)


# ════════════════════════════════════════════════════════════════════════════════
# 4. STREAMLIT UI — Layout & Interactivity
# ════════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
<style>
.stApp { background: #0F172A; }
.block-container { padding: 1.2rem 2rem 3rem; }

.siop-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #0F172A 60%, #1a1040 100%);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 12px;
    padding: 18px 28px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 16px;
}
.siop-title { font-size: 1.6rem; font-weight: 700; color: #F1F5F9; margin: 0; }
.siop-sub   { font-size: 0.82rem; color: #94A3B8; margin: 0; letter-spacing: 0.04em; }
.badge {
    background: rgba(245,158,11,0.15);
    border: 1px solid rgba(245,158,11,0.35);
    border-radius: 6px; padding: 3px 10px;
    color: #F59E0B; font-size: 0.75rem; font-weight: 600;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 18px;
}

[data-testid="stSidebar"] {
    background: #0B1120;
    border-right: 1px solid rgba(255,255,255,0.07);
}

.section-header {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.1em;
    color: #64748B; text-transform: uppercase;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 6px; margin: 1.4rem 0 0.8rem;
}

.exc-row      { border-radius: 8px; padding: 10px 16px; margin-bottom: 6px; }
.exc-critical { background: rgba(239,68,68,0.10); border-left: 3px solid #EF4444; }
.exc-warning  { background: rgba(245,158,11,0.10); border-left: 3px solid #F59E0B; }
.exc-clear    { background: rgba(16,185,129,0.10); border-left: 3px solid #10B981; }
</style>

<div class="siop-header">
    <div>
        <p class="siop-title">🏭 SIOP Digital Twin</p>
        <p class="siop-sub">Enterprise Knowledge Graph · Financial-First Supply Chain Planning</p>
    </div>
    <span class="badge">⚡ LIVE SCENARIO</span>
</div>
""", unsafe_allow_html=True)


def render_sidebar() -> ScenarioParams:
    """Sidebar scenario levers — returns a ScenarioParams dataclass instance."""
    with st.sidebar:
        st.markdown("## 🎛️ Scenario Levers")
        st.caption("Adjust parameters to model planning scenarios in real-time.")

        st.markdown('<div class="section-header">📈 Demand</div>', unsafe_allow_html=True)
        demand_shift = st.slider(
            "Global Demand Shift (%)", -40, 40, 0, 1,
            help="OEM pull-forward (+) or ramp-down (−). Applied to all SKUs.",
        )

        st.markdown('<div class="section-header">🛢️ Raw Material Costs</div>', unsafe_allow_html=True)
        rm_spike = st.slider(
            "Broad RM Cost Spike (%)", 0, 60, 0, 1,
            help="Applies to ALL components (commodity inflation).",
        )
        crit_rm_spike = st.slider(
            "Critical Component Spike (%)", 0, 80, 0, 1,
            help="Additional increase on Critical-risk parts only (chips, lithium, magnets).",
        )

        st.markdown('<div class="section-header">✈️ Freight & Logistics</div>', unsafe_allow_html=True)
        expedite_pct = st.slider(
            "Expedited Air Freight (% of volume)", 0, 50, 0, 1,
            help="Supply disruption response — shifts ocean → air freight lanes.",
        )

        st.markdown('<div class="section-header">🏭 Manufacturing Capacity</div>', unsafe_allow_html=True)
        ot_hours = st.number_input(
            "Overtime Hours Added / Month", 0, 400, 0, 20,
            help="Adds to net available production hours (weekend / extra shifts).",
        )
        scrap_delta = st.slider(
            "Additional Scrap Rate (%)", 0.0, 5.0, 0.0, 0.1,
            help="Quality excursion or new-part qualification yield loss.",
        )

        st.markdown('<div class="section-header">💲 Pricing</div>', unsafe_allow_html=True)
        asp_change = st.slider(
            "ASP Change — Pass-through (%)", -10, 10, 0, 1,
            help="Contractual price increases (+) or OEM rebate pressure (−).",
        )

        st.markdown('<div class="section-header">⚠️ Alert Thresholds</div>', unsafe_allow_html=True)
        margin_threshold = st.number_input(
            "Margin Alert Threshold (%)", 5.0, 40.0, 20.0, 1.0,
        )
        capacity_threshold = st.number_input(
            "Capacity Alert Threshold (%)", 70.0, 100.0, 95.0, 1.0,
        )

        st.markdown("---")
        if st.button("🔄 Reset to Baseline", use_container_width=True):
            st.rerun()

    return ScenarioParams(
        demand_shift_pct       = float(demand_shift),
        rm_cost_spike_pct      = float(rm_spike),
        critical_rm_spike_pct  = float(crit_rm_spike),
        expedite_freight_pct   = float(expedite_pct),
        capacity_add_hrs       = float(ot_hours),
        scrap_rate_delta       = float(scrap_delta),
        asp_change_pct         = float(asp_change),
        margin_alert_threshold = float(margin_threshold),
        capacity_alert_pct     = float(capacity_threshold),
    )


def render_kpi_strip(pl_bridge: Dict[str, float], capacity_df: pd.DataFrame):
    """Six headline KPI metric cards."""
    M = 1_000_000
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        delta_rev = pl_bridge["scenario_rev"] - pl_bridge["baseline_rev"]
        st.metric(
            "📦 Scenario Revenue",
            f"${pl_bridge['scenario_rev'] / M:.1f}M",
            f"{delta_rev / M:+.2f}M",
        )
    with c2:
        delta_gp = pl_bridge["scenario_gp"] - pl_bridge["baseline_gp"]
        st.metric(
            "💰 Scenario Gross Profit",
            f"${pl_bridge['scenario_gp'] / M:.1f}M",
            f"{delta_gp / M:+.2f}M",
        )
    with c3:
        delta_gm = pl_bridge["scenario_gm_pct"] - pl_bridge["baseline_gm_pct"]
        st.metric(
            "📊 Gross Margin %",
            f"{pl_bridge['scenario_gm_pct']:.1f}%",
            f"{delta_gm:+.1f} ppts",
        )
    with c4:
        max_util      = capacity_df["utilisation_pct"].max()
        n_constrained = int((capacity_df["utilisation_pct"] >= 95).sum())
        st.metric(
            "🏭 Peak Utilisation",
            f"{max_util:.1f}%",
            f"{n_constrained} month(s) constrained",
            delta_color="inverse" if n_constrained > 0 else "off",
        )
    with c5:
        st.metric(
            "📉 Baseline GM %",
            f"{pl_bridge['baseline_gm_pct']:.1f}%",
            "Reference Plan",
            delta_color="off",
        )
    with c6:
        st.metric(
            "⚡ GP Bridge Impact",
            f"{pl_bridge['gp_delta'] / M:+.2f}M",
            "vs Baseline",
            delta_color="normal" if pl_bridge["gp_delta"] >= 0 else "inverse",
        )


def render_exceptions(exceptions_df: pd.DataFrame):
    """Colour-coded exception alert cards with recommended actions."""
    st.markdown(
        '<div class="section-header">⚠️ Exception Management — Automated Plan Alerts</div>',
        unsafe_allow_html=True,
    )
    for _, row in exceptions_df.iterrows():
        sev = row["severity"]
        css = (
            "exc-clear"    if "CLEAR"    in sev else
            "exc-critical" if "CRITICAL" in sev else
            "exc-warning"
        )
        st.markdown(f"""
<div class="exc-row {css}">
    <strong>{row['severity']}</strong> &nbsp;|&nbsp;
    <strong>{row['category']}</strong>: {row['entity']} &nbsp;—&nbsp;
    {row['metric']}
    &nbsp;<span style="color:#94A3B8;">(Threshold: {row['threshold']})</span><br>
    <span style="font-size:0.82rem;color:#CBD5E1;">
        → Recommended Action: {row['action']}
    </span>
</div>
""", unsafe_allow_html=True)


def render_cost_detail_table(sc_costs: pd.DataFrame, margin_threshold: float):
    """SKU-level cost decomposition table with conditional margin highlighting."""
    st.markdown(
        '<div class="section-header">🔍 SKU Cost Decomposition</div>',
        unsafe_allow_html=True,
    )
    display = sc_costs[[
        "sku_name", "raw_material_cost", "scenario_rm_total",
        "direct_labor_cost", "overhead_cost",
        "scenario_cogs", "scenario_asp", "scenario_gm_pct", "gm_delta_ppts",
    ]].copy()
    display.columns = [
        "SKU", "Baseline RM ($)", "Scenario RM ($)",
        "Labor ($)", "Overhead ($)",
        "Scenario COGS ($)", "Scenario ASP ($)", "Scenario GM%", "GM Δ (ppts)",
    ]
    display = display.round(2)

    def highlight_gm(val):
        if isinstance(val, (int, float)) and val < margin_threshold:
            return "background-color: rgba(239,68,68,0.20); color: #FCA5A5;"
        return ""

    def highlight_delta(val):
        if isinstance(val, (int, float)):
            if val < -2:
                return "color: #FCA5A5;"
            if val > 0:
                return "color: #6EE7B7;"
        return ""

    styled = (
        display.style
               .map(highlight_gm,    subset=["Scenario GM%"])
               .map(highlight_delta, subset=["GM Δ (ppts)"])
               .format({"Scenario GM%": "{:.1f}%", "GM Δ (ppts)": "{:+.1f}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# 5. MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def main():
    # Session state counter (tracks re-runs for debugging)
    st.session_state.setdefault("run_count", 0)
    st.session_state["run_count"] += 1

    render_header()

    # ── Load master data (cached after first call) ────────────────────────────
    erp_data = generate_synthetic_erp_data()

    # ── Sidebar controls → scenario parameters ────────────────────────────────
    params = render_sidebar()

    # ── Full calculation run ──────────────────────────────────────────────────
    engine  = DigitalTwinEngine(erp_data, params)
    results = engine.run()

    sc_demand  = results["sc_demand"]
    sc_costs   = results["sc_costs"]
    capacity   = results["capacity"]
    pl_bridge  = results["pl_bridge"]
    inventory  = results["inventory"]
    exceptions = results["exceptions"]

    # ── KPI headline strip ────────────────────────────────────────────────────
    render_kpi_strip(pl_bridge, capacity)
    st.markdown("---")

    # ── Exception panel ───────────────────────────────────────────────────────
    render_exceptions(exceptions)

    # ── Row 1: Demand & Capacity ──────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📦 Volume & Capacity Analysis</div>',
        unsafe_allow_html=True,
    )
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(chart_demand_comparison(sc_demand), use_container_width=True)
    with col_r:
        st.plotly_chart(
            chart_capacity_utilisation(capacity, params.capacity_alert_pct),
            use_container_width=True,
        )

    # ── Row 2: P&L Waterfall & Margin by SKU ─────────────────────────────────
    st.markdown(
        '<div class="section-header">💰 Financial Impact Analysis</div>',
        unsafe_allow_html=True,
    )
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(chart_pl_waterfall(pl_bridge), use_container_width=True)
    with col_r:
        st.plotly_chart(
            chart_margin_by_sku(sc_costs, params.margin_alert_threshold),
            use_container_width=True,
        )

    # ── Row 3: Inventory & Working Capital ────────────────────────────────────
    st.markdown(
        '<div class="section-header">📊 Inventory & Working Capital</div>',
        unsafe_allow_html=True,
    )
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(chart_inventory_trend(inventory), use_container_width=True)
    with col_r:
        st.plotly_chart(chart_working_capital(inventory), use_container_width=True)

    # ── Capacity detail table ─────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">🏭 Monthly Capacity Detail</div>',
        unsafe_allow_html=True,
    )
    cap_display = capacity[[
        "month_label", "available_hours", "planned_maintenance_hrs",
        "ot_hours_available", "net_available_hours", "required_asm_hrs",
        "utilisation_pct", "capacity_status",
    ]].copy()
    cap_display.columns = [
        "Month", "Gross Hrs", "Maint Hrs", "OT Hrs",
        "Net Avail Hrs", "Required Hrs", "Utilisation %", "Status",
    ]
    st.dataframe(cap_display.round(1), use_container_width=True, hide_index=True)

    # ── SKU cost decomposition ────────────────────────────────────────────────
    render_cost_detail_table(sc_costs, params.margin_alert_threshold)

    # ── BOM supply risk explorer ──────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">🧩 BOM Supply Risk Explorer</div>',
        unsafe_allow_html=True,
    )
    bom = erp_data["bom"].copy()
    risk_icons = {"Critical": "🔴 Critical", "Standard": "🟢 Standard"}
    bom["Supply Risk"] = bom["supply_risk"].map(risk_icons)
    st.dataframe(
        bom[["sku", "component_name", "std_cost_usd", "origin", "Supply Risk"]]
           .rename(columns={
               "sku": "SKU",
               "component_name": "Component",
               "std_cost_usd": "Std Cost ($)",
               "origin": "Origin",
           }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"🏭 SIOP Digital Twin · Tier-1 Automotive Manufacturing · "
        f"Calc run #{st.session_state['run_count']} · "
        f"Pandas {pd.__version__} | NumPy {np.__version__}"
    )


if __name__ == "__main__":
    main()
