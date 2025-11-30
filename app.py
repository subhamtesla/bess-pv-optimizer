#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Streamlit dashboard for the BESS + PV optimizer (dynamic window & devices)

import streamlit as st
import pandas as pd
from datetime import datetime, time, date
from pathlib import Path
import requests

# Standard-library timezone (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
    SLC_TZ = ZoneInfo("America/Denver")
except Exception:
    SLC_TZ = None

# Import optimizer from the SAME folder as this app.py
from optimizer import run_optimization

st.set_page_config(page_title="BESS + PV Optimizer", layout="wide")
st.title("⚡ BESS + PV Scheduling — Interactive Dashboard")


# ---------------------------------------------------------
# Helper: get current time & temperature in Salt Lake City
# ---------------------------------------------------------
def get_current_slc_conditions():
    """
    Returns (now_dt, temp_c, temp_f)

    If anything fails, temp_c and temp_f are None.
    """
    # Current time in SLC (fallback to UTC if tz not available)
    if SLC_TZ is not None:
        now = datetime.now(SLC_TZ)
    else:
        now = datetime.utcnow()

    temp_c = None
    temp_f = None

    try:
        # Salt Lake City approx: 40.7608 N, -111.8910 W
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=40.7608&longitude=-111.8910&current_weather=true"
        )
        r = requests.get(url, timeout=5)
        if r.ok:
            data = r.json()
            cw = data.get("current_weather", {})
            t_c = cw.get("temperature")
            if t_c is not None:
                temp_c = float(t_c)
                temp_f = temp_c * 9.0 / 5.0 + 32.0
    except Exception:
        # If the API fails for any reason, just leave temps as None
        pass

    return now, temp_c, temp_f


# ---------------------------------------------------------
# 1) House ID gate (sidebar)
# ---------------------------------------------------------
with st.sidebar:
    st.header("House selection")
    house_id = st.text_input(
        "House ID",
        value="",
        type="password",          # behaves like a password field
        help="Enter your House ID to access the dashboard."
    )

# If ID is wrong or empty, stop here
if house_id != "01592915":
    st.warning("Please enter the valid House ID to access the dashboard.")
    st.stop()


# ---------------------------------------------------------
# 2) Top “status bar”: current time & temperature in SLC
# ---------------------------------------------------------
now_slc, temp_c, temp_f = get_current_slc_conditions()
kpi_col1, kpi_col2 = st.columns(2)

with kpi_col1:
    time_label = (
        "Current time (Salt Lake City)"
        if SLC_TZ is not None
        else "Current time (UTC)"
    )
    st.metric(time_label, now_slc.strftime("%Y-%m-%d %H:%M:%S"))

with kpi_col2:
    if temp_f is not None and temp_c is not None:
        st.metric(
            "Current temperature (Salt Lake City)",
            f"{temp_f:.1f} °F",
            f"{temp_c:.1f} °C",
        )
    else:
        st.metric("Current temperature (Salt Lake City)", "N/A")

st.markdown("---")


# ---------------------------------------------------------
# 3) Sidebar: global settings
# ---------------------------------------------------------
with st.sidebar:
    st.header("Global settings")

    data_file = st.text_input(
        "Excel data file",
        value="data/Load_price_2025.xlsx",
        help="Must have columns: datetime, Solar, Load, Price",
    )

    start_date = st.date_input("Start date", date(2025, 11, 9))
    start_time = st.time_input("Start time", time(4, 0))
    end_date   = st.date_input("End date", date(2025, 11, 15))
    end_time   = st.time_input("End time", time(4, 0))

    grid_max   = st.number_input("Grid max power (kW)", 0.0, 50.0, 12.0, 0.1)
    solar_mult = st.number_input("PV capacity multiplier", 0.0, 20.0, 1.37, 0.01)

    bess_p     = st.number_input("BESS charge/discharge (kW)", 0.0, 50.0, 4.44, 0.01)
    bess_e     = st.number_input("BESS energy (kWh)", 0.1, 100.0, 4.44, 0.01)
    bess_soc_lo= st.slider("BESS min SOC", 0.0, 1.0, 0.20, 0.05)
    bess_soc_hi= st.slider("BESS max SOC", 0.0, 1.0, 0.90, 0.05)
    final_soc_tol = st.number_input(
        "Final SOC tolerance (kWh)", 0.0, 1.0, 0.001, 0.001
    )


# ---------------------------------------------------------
# 4) Main layout: left = optimization/results, right = devices
# ---------------------------------------------------------
left_col, right_col = st.columns([3, 1])


# ---------- RIGHT: Flexible devices (in collapsible panels) ----------
with right_col:
    st.subheader("Flexible devices")

    st.caption(
        "Click on a device to configure its power, runtime, "
        "and available time windows."
    )

    def _parse_window_lines(txt: str):
        """Parse text area lines as '(start, end)' tuples."""
        win = []
        for line in (txt or "").splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                win.append((parts[0], parts[1]))
        return win

    # EV
    with st.expander("EV charger (Level-2)", expanded=True):
        ev_on   = st.checkbox("Include EV", True, key="ev_on")
        ev_p    = st.number_input(
            "EV power (kW)", 0.0, 20.0, 7.7, 0.1, key="ev_p"
        )
        ev_h    = st.number_input(
            "EV hours", 0.5, 24.0, 6.0, 0.5, key="ev_h"
        )
        ev_win  = st.text_area(
            "EV windows (one per line: 'MM-DD-YYYY HH:MM:SS, MM-DD-YYYY HH:MM:SS')",
            "11-11-2025 21:00:00, 11-12-2025 07:00:00",
            key="ev_win",
        )

    # Dishwasher
    with st.expander("Dishwasher", expanded=False):
        dw_on  = st.checkbox("Include Dishwasher", True, key="dw_on")
        dw_p   = st.number_input(
            "Dishwasher power (kW)", 0.0, 5.0, 1.2, 0.1, key="dw_p"
        )
        dw_h   = st.number_input(
            "Dishwasher hours", 0.5, 6.0, 1.5, 0.5, key="dw_h"
        )
        dw_win = st.text_area(
            "Dishwasher windows",
            "11-10-2025 18:00:00, 11-10-2025 23:00:00",
            key="dw_win",
        )

    # Dryer
    with st.expander("Clothes dryer", expanded=False):
        dr_on  = st.checkbox("Include Dryer", True, key="dr_on")
        dr_p   = st.number_input(
            "Dryer power (kW)", 0.0, 10.0, 4.5, 0.1, key="dr_p"
        )
        dr_h   = st.number_input(
            "Dryer hours", 0.25, 4.0, 0.75, 0.25, key="dr_h"
        )
        dr_win = st.text_area(
            "Dryer windows",
            "11-13-2025 13:00:00, 11-13-2025 19:00:00",
            key="dr_win",
        )

    # Washer
    with st.expander("Washing machine", expanded=False):
        wa_on  = st.checkbox("Include Washer", True, key="wa_on")
        wa_p   = st.number_input(
            "Washer power (kW)", 0.0, 5.0, 0.5, 0.1, key="wa_p"
        )
        wa_h   = st.number_input(
            "Washer hours", 0.25, 4.0, 1.0, 0.25, key="wa_h"
        )
        wa_win = st.text_area(
            "Washer windows",
            "11-13-2025 12:00:00, 11-13-2025 17:00:00",
            key="wa_win",
        )

    st.info(
        "Each line in a window field should look like:\n\n"
        "`MM-DD-YYYY HH:MM:SS, MM-DD-YYYY HH:MM:SS`"
    )


# ---------- LEFT: Optimization + results ----------
with left_col:
    st.subheader("Optimization & Results")

    run_clicked = st.button("Optimize schedule", type="primary")

    if run_clicked:
        START = datetime.combine(start_date, start_time)
        END   = datetime.combine(end_date, end_time)

        # Build list of active devices
        devices = []
        if ev_on:
            devices.append({
                "name": "EV",
                "P_kw": ev_p,
                "hours": ev_h,
                "windows": _parse_window_lines(ev_win),
            })
        if dr_on:
            devices.append({
                "name": "Dryer",
                "P_kw": dr_p,
                "hours": dr_h,
                "windows": _parse_window_lines(dr_win),
            })
        if dw_on:
            devices.append({
                "name": "Dishwasher",
                "P_kw": dw_p,
                "hours": dw_h,
                "windows": _parse_window_lines(dw_win),
            })
        if wa_on:
            devices.append({
                "name": "WashingMachine",
                "P_kw": wa_p,
                "hours": wa_h,
                "windows": _parse_window_lines(wa_win),
            })

        # Basic checks
        if not Path(data_file).exists():
            st.error(f"Data file not found: {data_file}")
        elif START >= END:
            st.error("End time must be after Start time.")
        elif not devices:
            st.error("Please include at least one device.")
        else:
            with st.spinner("Running optimization..."):
                # run_optimization returns: results_df, schedule_df, pngs, xlsx, summary_text
                results_df, schedule_df, pngs, xlsx, summary_text = run_optimization(
                    DATA_FILE=data_file,
                    START_TIME=START,
                    END_TIME=END,
                    DEVICES=devices,
                    GRID_MAX=grid_max,
                    SOLAR_CAPACITY=solar_mult,
                    BESS_P=bess_p,
                    BESS_E=bess_e,
                    BESS_SOC_LO=bess_soc_lo,
                    BESS_SOC_HI=bess_soc_hi,
                    FINAL_SOC_TOL=final_soc_tol,
                )

            if results_df is None or results_df.empty:
                st.error(
                    "Model infeasible or no data in the selected window.\n"
                    "Try raising Grid max power, lowering device power/hours, "
                    "or relaxing Final SOC tolerance."
                )
            else:
                st.success("Optimization complete.")

                # Small KPIs from results (optional simple ones)
                k1, k2, k3 = st.columns(3)
                with k1:
                    total_load = float(results_df["Total_Load"].sum()) * (
                        (results_df["Time"].iloc[1] - results_df["Time"].iloc[0])
                        .total_seconds() / 3600.0
                    ) if len(results_df) > 1 else 0.0
                    st.metric("Total Load Demand (kWh)", f"{total_load:.1f}")
                with k2:
                    soc_start = results_df["Battery_SOC"].iloc[0]
                    soc_end   = results_df["Battery_SOC"].iloc[-1]
                    st.metric("Battery SOC (start → end)", f"{soc_end:.1f} %", f"{soc_end - soc_start:+.1f} %")
                with k3:
                    # If summary_text has a "TOTAL ENERGY COST" line, this is nicer,
                    # but we just show that the summary is available.
                    st.metric("Summary available", "Yes")

                st.markdown("### Timeseries (preview)")
                st.dataframe(results_df.head(200))

                st.markdown("### Device schedule")
                st.dataframe(schedule_df)

                # Plots
                if pngs:
                    st.markdown("### Plots")
                    for p in pngs:
                        if p and Path(p).exists():
                            st.image(str(p), use_column_width=True)

                # Download Excel
                if xlsx and Path(xlsx).exists():
                    st.download_button(
                        "Download Excel (Timeseries + Schedule + Summary)",
                        data=open(xlsx, "rb").read(),
                        file_name=Path(xlsx).name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                # Detailed summary (text block)
                if summary_text:
                    st.markdown("### Detailed Power & Energy Summary")
                    st.code(summary_text)
