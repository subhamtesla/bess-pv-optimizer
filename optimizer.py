#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================================================
# BESS + PV Scheduling with PuLP (function form for Streamlit dashboard)
# - START/END and DEVICES are dashboard-driven (can still fall back)
# - Each flexible device runs a single contiguous block within its windows
# - Saves a dashboard PNG + one PNG per device; exports Excel
# - Tightened discharge-start detection; min-run enforced
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plotting for Streamlit/server
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import pulp
from pathlib import Path
from datetime import datetime

plt.rcParams.update({'font.family': 'Times New Roman'})

# -----------------------------
# Helpers
# -----------------------------
def _parse_dt(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, errors='raise', dayfirst=False)

def _hours_to_steps(h: float, step_hours: float) -> int:
    return max(1, int(round(h / step_hours)))

def _build_allowed_start_mask_from_windows(
    timestamps: pd.Series,
    windows: list,
    duration_steps: int,
    minutes_per_step: int
) -> np.ndarray:
    """
    For a list of (start_str, end_str) windows, return a boolean mask 'allowed_start'
    over timeline indices where a device is allowed to START such that its full block
    (duration_steps) finishes before the window end.

    A start at index t is allowed if:
        timestamps[t] >= window_start
    and timestamps[t + duration_steps - 1] + 1 step <= window_end.
    """
    n = len(timestamps)
    if not windows:
        return np.zeros(n, dtype=bool)

    ts = timestamps.to_numpy()
    allowed = np.zeros(n, dtype=bool)

    for (ws, we) in windows:
        wstart = _parse_dt(ws)
        wend   = _parse_dt(we)

        for t in range(n):
            if t + duration_steps > n:
                break
            t_start = ts[t]
            t_end   = ts[min(t + duration_steps - 1, n - 1)] + pd.to_timedelta(minutes_per_step, unit="m")
            if (t_start >= wstart) and (t_end <= wend):
                allowed[t] = True

    return allowed

def _format_time_axis(ax):
    """Improve time axis readability for long windows."""
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    for label in ax.get_xticklabels():
        label.set_rotation(0)

# ---------------------------------------------------------------------
# MAIN API (called by Streamlit)
# ---------------------------------------------------------------------
def run_optimization(
    DATA_FILE: str,
    # Window & devices (dashboard-driven). If None, sensible fallbacks below.
    START_TIME: datetime = None,
    END_TIME: datetime = None,
    DEVICES: list = None,  # list of dicts: {"name","P_kw","hours","windows":[(start,end), ...]}

    # Power-system parameters
    GRID_MAX: float = 12.0,
    SOLAR_CAPACITY: float = 1.37,
    BESS_P: float = 4.44,         # charge/discharge kW (symmetric)
    BESS_E: float = 4.44,         # kWh
    BESS_SOC_LO: float = 0.20,    # [0..1]
    BESS_SOC_HI: float = 0.90,    # [0..1]
    FINAL_SOC_TOL: float = 1e-6,  # kWh
    PRICE_DIVISOR: float = 100.0, # cents/kWh -> $/kWh
    SOLAR_IN_WATTS: bool = True,
    bess_charge_efficiency: float = 0.95,
    bess_discharge_efficiency: float = 0.95,
    bess_om_cost: float = 0.046,      # $/kWh on discharge
    min_runtime_discharge: int = 1,   # timesteps
    min_discharge_power_ratio: float = 0.20,
    pv_energy_cost: float = 0.01,     # $/kWh PV O&M proxy
    load_curtail_cost: float = 2.0,   # $/kWh penalty
    CBC_TIME_LIMIT_S: int = 600,
    CBC_REL_GAP: float = 0.01,
    PLOTS_DIR: Path = None
):
    """
    Returns:
        results_df, schedule_df, png_paths(list[str]), excel_path(str)
        If infeasible: returns (empty_df, minimal_schedule_df, [], None)
    """
    if PLOTS_DIR is None:
        PLOTS_DIR = Path.cwd()

    # Fallback fixed window if dashboard didn't pass any
    if START_TIME is None:
        START_TIME = datetime(2025, 11, 9, 4, 0, 0)
    if END_TIME is None:
        END_TIME   = datetime(2025, 11, 15, 4, 0, 0)

    # Fallback devices (the four you asked for) if dashboard didn't pass any
    if DEVICES is None:
        DEVICES = [
          {"name":"EV","P_kw":7.7,"hours":6.0,"windows":[("11-11-2025 21:00:00","11-12-2025 07:00:00")]},
          {"name":"Dryer","P_kw":4.5,"hours":0.75,"windows":[("11-13-2025 13:00:00","11-13-2025 19:00:00")]},
          {"name":"Dishwasher","P_kw":1.2,"hours":1.5,"windows":[("11-10-2025 18:00:00","11-10-2025 23:00:00")]},
          {"name":"WashingMachine","P_kw":0.5,"hours":1.0,"windows":[("11-13-2025 12:00:00","11-13-2025 17:00:00")]},
        ]

    # -----------------------------
    # Load & preprocess data
    # -----------------------------
    df = pd.read_excel(
        DATA_FILE,
        usecols=['datetime', 'Solar', 'Load', 'Price'],
        engine='openpyxl'
    )
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['Solar']    = pd.to_numeric(df['Solar'], errors='coerce')
    df['Load']     = pd.to_numeric(df['Load'], errors='coerce')
    df['Price']    = pd.to_numeric(df['Price'], errors='coerce')
    df = (
        df.dropna(subset=['datetime', 'Solar', 'Load', 'Price'])
          .sort_values('datetime')
          .drop_duplicates('datetime')
          .reset_index(drop=True)
    )

    # Filter to window (END exclusive)
    start_ts = pd.to_datetime(START_TIME)
    end_ts   = pd.to_datetime(END_TIME)
    mask = (df['datetime'] >= start_ts) & (df['datetime'] < end_ts)
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        empty_results = pd.DataFrame()
        empty_sched = pd.DataFrame([{"Device":"(no data rows in window)", "Power_kW":0, "Duration_steps":0,
                                     "Duration_min":0, "Start_time":pd.NaT, "End_time":pd.NaT, "Energy_kWh":0}])
        return empty_results, empty_sched, [], None

    # Step size detection
    if len(df) >= 2:
        dt_minutes = df['datetime'].diff().dropna().dt.total_seconds().median() / 60.0
        step_size  = dt_minutes / 60.0  # hours/step
    else:
        dt_minutes = 10
        step_size  = 10 / 60.0
    minutes_per_step = int(round(step_size * 60))

    # Forecast arrays
    load_forecast  = df['Load'].to_numpy(dtype=float)                             # kW
    price_forecast = (df['Price'] / PRICE_DIVISOR).to_numpy(dtype=float)         # $/kWh
    solar_forecast = (df['Solar']/1000.0 if SOLAR_IN_WATTS else df['Solar']).to_numpy(dtype=float)  # kW

    time_horizon = len(df)
    t_index = np.arange(time_horizon)

    # -----------------------------
    # Build flexible-device structs from DEVICES
    # -----------------------------
    devices = []
    for dev in DEVICES:
        devices.append({
            "name": dev["name"],
            "P_kw": float(dev["P_kw"]),
            "hours": float(dev["hours"]),
            "windows": list(dev["windows"])
        })

    # Convert hours -> steps, and precompute per-device allowed start masks
    for d in devices:
        d["D"] = _hours_to_steps(d["hours"], step_size)
        d["allowed_start"] = _build_allowed_start_mask_from_windows(
            df['datetime'], d["windows"], d["D"], minutes_per_step
        )

    # -----------------------------
    # Model setup
    # -----------------------------
    bess_min_level = BESS_SOC_LO * BESS_E
    bess_max_level = BESS_SOC_HI * BESS_E
    inv_eta_dis = 1.0 / float(bess_discharge_efficiency)
    eta_chg     = float(bess_charge_efficiency)

    model = pulp.LpProblem("BESS_Optimization_with_FlexibleLoads_ExactWindows", pulp.LpMinimize)
    T = range(time_horizon)

    # Decision variables
    P_grid       = pulp.LpVariable.dicts("P_grid", T, lowBound=-GRID_MAX, upBound=GRID_MAX, cat=pulp.LpContinuous)
    P_charge     = pulp.LpVariable.dicts("P_charge", T, lowBound=0, upBound=BESS_P, cat=pulp.LpContinuous)
    P_discharge  = pulp.LpVariable.dicts("P_discharge", T, lowBound=0, upBound=BESS_P, cat=pulp.LpContinuous)
    E_battery    = pulp.LpVariable.dicts("E_battery", T, lowBound=bess_min_level, upBound=bess_max_level, cat=pulp.LpContinuous)
    P_load_curt  = pulp.LpVariable.dicts("P_load_curt",  T, lowBound=0, cat=pulp.LpContinuous)
    P_solar_curt = pulp.LpVariable.dicts("P_solar_curt", T, lowBound=0, cat=pulp.LpContinuous)
    P_pv_used    = pulp.LpVariable.dicts("P_pv_used", T, lowBound=0, cat=pulp.LpContinuous)
    z_bess       = pulp.LpVariable.dicts("z_bess", T, lowBound=0, upBound=1, cat=pulp.LpBinary)
    discharge_start = pulp.LpVariable.dicts("discharge_start", T, lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Flexible devices: start binaries only
    job_s_vars = []  # (dev_dict, s)
    for dev in devices:
        s = pulp.LpVariable.dicts(f"{dev['name']}_start", T, lowBound=0, upBound=1, cat=pulp.LpBinary)
        job_s_vars.append((dev, s))

    def device_power_at_t(dev, s, t):
        """Device power at time t = P * sum_{j=max(0,t-D+1)}^{t} s[j] (contiguous block)."""
        D = dev["D"]
        left = max(0, t - D + 1)
        return dev["P_kw"] * pulp.lpSum([s[j] for j in range(left, t + 1)])

    # Constraints
    # (1) Power balance
    for t in T:
        dev_load_t = 0
        for dev, s in job_s_vars:
            dev_load_t += device_power_at_t(dev, s, t)
        load_served = load_forecast[t] + dev_load_t - P_load_curt[t]
        model += P_pv_used[t] + P_discharge[t] + P_grid[t] == load_served + P_charge[t], f"power_balance_{t}"

    # (2) PV balance
    for t in T:
        solar_available = solar_forecast[t] * SOLAR_CAPACITY
        model += P_pv_used[t] + P_solar_curt[t] == solar_available, f"pv_balance_{t}"

    # (3) BESS dynamics
    initial_battery_soc   = 0.5
    initial_battery_level = initial_battery_soc * BESS_E
    for t in T:
        charge_term    = step_size * (P_charge[t] * eta_chg)
        discharge_term = step_size * (P_discharge[t] * inv_eta_dis)
        if t == 0:
            model += E_battery[t] == initial_battery_level + (charge_term - discharge_term), f"battery_dyn_{t}"
        else:
            model += E_battery[t] == E_battery[t-1] + (charge_term - discharge_term), f"battery_dyn_{t}"

    # (4) Charge/discharge logic
    for t in T:
        model += P_charge[t]    <= BESS_P * (1 - z_bess[t]), f"charge_limit_{t}"
        model += P_discharge[t] >= min_discharge_power_ratio * BESS_P * z_bess[t], f"dis_min_{t}"
        model += P_discharge[t] <= BESS_P * z_bess[t], f"dis_max_{t}"

    # (5) Discharge start + minimum runtime (tightened detection)
    for t in T:
        if t == 0:
            model += discharge_start[t] >= z_bess[t],                          f"start_ge_z_init_{t}"
            model += discharge_start[t] <= z_bess[t],                          f"start_le_z_init_{t}"
        else:
            model += discharge_start[t] >= z_bess[t] - z_bess[t-1],            f"start_ge_diff_{t}"
            model += discharge_start[t] <= z_bess[t],                          f"start_le_z_{t}"
            model += discharge_start[t] <= 1 - z_bess[t-1],                    f"start_zero_if_prev_on_{t}"

    for t in range(time_horizon - min_runtime_discharge + 1):
        rhs = min_runtime_discharge * discharge_start[t]
        model += pulp.lpSum([z_bess[t + k] for k in range(min_runtime_discharge)]) >= rhs, f"min_runtime_{t}"

    for t in range(max(0, time_horizon - min_runtime_discharge + 1), time_horizon):
        remaining = time_horizon - t
        model += pulp.lpSum([z_bess[t + k] for k in range(remaining)]) >= remaining * discharge_start[t], f"min_runtime_tail_{t}"

    # (6) End SOC ≈ initial
    model += E_battery[time_horizon - 1] >= initial_battery_level - FINAL_SOC_TOL, "final_soc_lo"
    model += E_battery[time_horizon - 1] <= initial_battery_level + FINAL_SOC_TOL, "final_soc_hi"

    # (7) Curtailment bounds
    for t in T:
        model += P_load_curt[t]  <= load_forecast[t], f"load_curt_limit_{t}"
        model += P_solar_curt[t] >= 0,                f"solar_curt_ge0_{t}"

    # (8) Device rules: allowed start mask + one start
    for dev, s in job_s_vars:
        name = dev["name"]
        D    = dev["D"]
        allowed = dev["allowed_start"]  # boolean array

        # Allow starts only where mask is True
        for t in T:
            if not bool(allowed[t]):
                model += s[t] == 0, f"{name}_forbid_start_{t}"

        # Forbid starts that overrun the horizon
        for t in range(time_horizon - D + 1, time_horizon):
            model += s[t] == 0, f"{name}_no_start_too_late_{t}"

        # Exactly one start
        model += pulp.lpSum([s[t] for t in T]) == 1, f"{name}_one_start"

    # Objective: energy cost + tiny tie-break to prefer earlier starts
    energy_cost = pulp.lpSum([
        step_size * price_forecast[t] * P_grid[t] +          # import cost; export negative
        step_size * pv_energy_cost * P_pv_used[t] +          # PV O&M proxy
        step_size * load_curtail_cost * P_load_curt[t] +     # curtailment penalty
        step_size * bess_om_cost * P_discharge[t]            # BESS O&M on discharge
        for t in T
    ])
    avg_price = float(np.mean(price_forecast)) if len(price_forecast) else 0.2
    EPS = 1e-6 * max(avg_price, 0.01)  # harmless tie-breaker
    early_start_penalty = 0
    for dev, s in job_s_vars:
        early_start_penalty += pulp.lpSum([EPS * t_index[t] * s[t] for t in T])
    model += energy_cost + early_start_penalty

    # -----------------------------
    # Solve
    # -----------------------------
    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=CBC_TIME_LIMIT_S, gapRel=CBC_REL_GAP)
    _ = model.solve(solver)
    status_str = pulp.LpStatus.get(model.status, "Unknown")

    ok_statuses = {'Optimal', 'Integer Feasible', 'Feasible'}
    if status_str not in ok_statuses:
        empty_results = pd.DataFrame()
        empty_sched = pd.DataFrame([{"Device":"(infeasible)", "Power_kW":0, "Duration_steps":0,
                                     "Duration_min":0, "Start_time":pd.NaT, "End_time":pd.NaT, "Energy_kWh":0}])
        return empty_results, empty_sched, [], None

    # -----------------------------
    # Results
    # -----------------------------
    device_cols = {}
    dev_total_load = np.zeros(time_horizon, dtype=float)
    schedule_rows = []
    for dev, s in job_s_vars:
        name, P, D = dev["name"], dev["P_kw"], dev["D"]
        s_vals = np.array([pulp.value(s[t]) for t in T], dtype=float)
        on_vals = np.zeros(time_horizon, dtype=float)
        for t in range(time_horizon):
            left = max(0, t - D + 1)
            on_vals[t] = s_vals[left:t+1].sum()
        on_vals = np.clip(on_vals, 0, 1)
        dev_load = P * on_vals
        dev_total_load += dev_load
        device_cols[f"{name}_Start"] = s_vals
        device_cols[f"{name}_On"] = on_vals
        # Schedule row
        if s_vals.sum() > 0.5:
            t0 = int(np.argmax(s_vals))
            t1 = min(t0 + D, time_horizon)
            start_time = df["datetime"].iloc[t0]
            end_time   = df["datetime"].iloc[t1 - 1] + pd.to_timedelta(minutes_per_step, unit="m")
            energy_kwh = P * (D * minutes_per_step / 60.0)
            schedule_rows.append({
                "Device": name,
                "Power_kW": P,
                "Duration_steps": D,
                "Duration_min": D * minutes_per_step,
                "Start_time": start_time,
                "End_time": end_time,
                "Energy_kWh": energy_kwh
            })

    results = {
        "Time":              [df["datetime"].iloc[t] for t in T],
        "Grid_Power":        [pulp.value(P_grid[t]) for t in T],
        "Charge_Power":      [pulp.value(P_charge[t]) for t in T],
        "Discharge_Power":   [pulp.value(P_discharge[t]) for t in T],
        "Battery_Level":     [pulp.value(E_battery[t]) for t in T],
        "Battery_SOC":       [pulp.value(E_battery[t]) / BESS_E * 100 for t in T],
        "Load_Curtailment":  [pulp.value(P_load_curt[t]) for t in T],
        "Solar_Curtailment": [pulp.value(P_solar_curt[t]) for t in T],
        "PV_Used":           [pulp.value(P_pv_used[t]) for t in T],
        "Base_Load":         load_forecast,
        "Device_Load":       dev_total_load,
        "Total_Load":        load_forecast + dev_total_load,
        "Solar_Available":   solar_forecast * SOLAR_CAPACITY,
        "Electricity_Price": price_forecast,
        "Discharge_Mode":    [pulp.value(z_bess[t]) for t in T],
        "Discharge_Start":   [pulp.value(discharge_start[t]) for t in T],
    }
    results.update(device_cols)
    results_df = pd.DataFrame(results)

    # Schedule df
    schedule_df = (pd.DataFrame(schedule_rows) if schedule_rows else
                   pd.DataFrame([{"Device": "(none)", "Power_kW": 0, "Duration_steps": 0,
                                  "Duration_min": 0, "Start_time": np.nan, "End_time": np.nan, "Energy_kWh": 0}]))

    # -----------------------------
    # SHOW & SAVE plots
    # -----------------------------
    out_base = f"BESS_Results_{start_ts:%Y%m%d_%H%M}_{end_ts:%Y%m%d_%H%M}"
    png_dashboard = PLOTS_DIR / f"{out_base}_dashboard.png"
    png_devplots  = {d["name"]: PLOTS_DIR / f"{out_base}_{d['name'].replace(' ', '_')}_schedule.png" for d in devices}

    x = results_df["Time"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1) Power flows
    axes[0, 0].plot(x, results_df["Grid_Power"], label="Grid Power", linewidth=2)
    axes[0, 0].plot(x, results_df["Charge_Power"], label="Battery Charge", linewidth=2)
    axes[0, 0].plot(x, results_df["Discharge_Power"], label="Battery Discharge", linewidth=2)
    axes[0, 0].plot(x, results_df["PV_Used"], label="PV Used", linewidth=2)
    axes[0, 0].plot(x, results_df["Solar_Available"], label="Solar Available", linewidth=2, linestyle="--", alpha=0.7)
    axes[0, 0].plot(x, results_df["Base_Load"], label="Base Load", linewidth=2)
    axes[0, 0].plot(x, results_df["Total_Load"], label="Total Load (with devices)", linewidth=2)
    axes[0, 0].axhline(y=0, linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Power Flows")
    axes[0, 0].set_ylabel("Power [kW]")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    _format_time_axis(axes[0, 0])

    # 2) Battery level and SOC
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(x, results_df["Battery_Level"], linewidth=2, label="Battery Level")
    ax2.axhline(y=bess_min_level, linestyle="--", label="Min Level", alpha=0.7)
    ax2.axhline(y=bess_max_level, linestyle="--", label="Max Level", alpha=0.7)
    ax2.set_ylabel("Energy [kWh]")
    ax2_twin.plot(x, results_df["Battery_SOC"], linewidth=2, linestyle=":", label="SOC %")
    ax2_twin.set_ylabel("State of Charge [%]")
    ax2_twin.set_ylim(0, 100)
    lines = line1 + ax2.lines[1:3] + ax2_twin.lines
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left")
    ax2.set_title("BESS Energy Level & State of Charge")
    ax2.grid(True, alpha=0.3)
    _format_time_axis(ax2)

    # 3) Load split
    axes[1, 0].plot(x, results_df["Base_Load"], label="Base Load", linewidth=2)
    axes[1, 0].plot(x, results_df["Device_Load"], label="Device Load", linewidth=2)
    axes[1, 0].plot(x, results_df["Total_Load"], label="Total Load", linewidth=2)
    axes[1, 0].set_title("Base vs Device Load")
    axes[1, 0].set_ylabel("Power [kW]")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    _format_time_axis(axes[1, 0])

    # 4) Electricity price
    axes[1, 1].plot(x, results_df["Electricity_Price"], linewidth=2)
    axes[1, 1].set_title("Electricity Price")
    axes[1, 1].set_ylabel("Price [$ / kWh]")
    axes[1, 1].grid(True, alpha=0.3)
    _format_time_axis(axes[1, 1])

    # # 5) BESS power
    # axes[2, 0].plot(x, results_df["Discharge_Power"], linewidth=2, label="Discharge")
    # axes[2, 0].plot(x, -results_df["Charge_Power"], linewidth=2, label="Charge (negative)")
    # axes[2, 0].axhline(y=0, linestyle="--", alpha=0.5)
    # axes[2, 0].set_title("BESS Power (+ = Discharge, - = Charge)")
    # axes[2, 0].set_ylabel("Power [kW]")
    # axes[2, 0].legend()
    # axes[2, 0].grid(True, alpha=0.3)
    # _format_time_axis(axes[2, 0])

    # # 6) Device starts
    # axes[2, 1].plot(x, results_df["Discharge_Mode"], label="BESS Discharge Mode", linewidth=2)
    # y_mark = 0.2
    # for dev, s in job_s_vars:
    #     s_vals = np.array([pulp.value(s[t]) for t in T])
    #     if s_vals.sum() > 0.5:
    #         t0 = int(np.argmax(s_vals))
    #         axes[2, 1].scatter([x.iloc[t0]], [y_mark], label=f"{dev['name']} start", s=80, marker="s")
    #         y_mark += 0.2
    # axes[2, 1].set_title("BESS Discharge & Device Start Times")
    # axes[2, 1].set_ylabel("Binary Status")
    # axes[2, 1].set_ylim(-0.1, 1.1)
    # axes[2, 1].legend()
    # axes[2, 1].grid(True, alpha=0.3)
    # _format_time_axis(axes[2, 1])

    plt.tight_layout()
    plt.savefig(png_dashboard, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Separate schedule plots for each device
    for dev, _s in job_s_vars:
        col = f"{dev['name']}_On"
        if col in results_df.columns:
            fig2, axd = plt.subplots(1, 1, figsize=(14, 3.5))
            axd.step(results_df["Time"], results_df[col], where='post', linewidth=2)
            axd.set_ylim(-0.1, 1.1)
            axd.set_yticks([0, 1])
            axd.set_title(f"{dev['name']} On/Off Schedule (1=ON)")
            axd.set_xlabel("Time")
            axd.set_ylabel("On/Off")
            axd.grid(True, alpha=0.3)
            _format_time_axis(axd)
            plt.tight_layout()
            out_png = png_devplots[dev["name"]]
            plt.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close(fig2)

    # Save Excel (2 sheets)
    out_xlsx  = PLOTS_DIR / f"{out_base}.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        results_df.to_excel(xw, index=False, sheet_name="Timeseries")
        schedule_df.to_excel(xw, index=False, sheet_name="Schedule")

    # Return paths for UI
    png_paths = [png_dashboard] + [png_devplots[d["name"]] for d in devices]
    return results_df, schedule_df, [str(p) for p in png_paths], str(out_xlsx)


# ---------------------------------------------------------------------
# Optional: quick CLI test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    
    DATA_FILE = "/home/ssingh/solar/forecasting/salt_lake_data/Load_price_2025.xlsx"
    results_df, schedule_df, pngs, xlsx = run_optimization(DATA_FILE=DATA_FILE)
    print("PNG files:", pngs)
    print("Excel file:", xlsx)
