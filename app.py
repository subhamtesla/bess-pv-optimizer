#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Streamlit dashboard for the BESS + PV optimizer (dynamic window & devices)

import streamlit as st
import pandas as pd
from datetime import datetime, time, date
from pathlib import Path

# Import optimizer from the SAME folder as this app.py
from optimizer import run_optimization

st.set_page_config(page_title="BESS + PV Optimizer", layout="wide")
st.title("⚡ BESS + PV Scheduling — Interactive Dashboard")

with st.sidebar:
    st.header("Global")
    default_path = "data/Load_price_2025.xlsx"
    data_file = st.text_input(
        "Excel data file",
        value=default_path,
        help="Must have columns: datetime, Solar, Load, Price"
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
    final_soc_tol = st.number_input("Final SOC tolerance (kWh)", 0.0, 1.0, 0.001, 0.001)

st.subheader("Flexible devices")
col1, col2 = st.columns(2)

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

with col1:
    st.markdown("**EV charger (Level-2)**")
    ev_on   = st.checkbox("Include EV", True)
    ev_p    = st.number_input("EV power (kW)", 0.0, 20.0, 7.7, 0.1, key="evp")
    ev_h    = st.number_input("EV hours", 0.5, 24.0, 6.0, 0.5, key="evh")
    ev_win  = st.text_area(
        "EV windows (one per line: 'MM-DD-YYYY HH:MM:SS, MM-DD-YYYY HH:MM:SS')",
        "11-11-2025 21:00:00, 11-12-2025 07:00:00"
    )

    st.markdown("---")
    st.markdown("**Dishwasher**")
    dw_on  = st.checkbox("Include Dishwasher", True)
    dw_p   = st.number_input("Dishwasher power (kW)", 0.0, 5.0, 1.2, 0.1)
    dw_h   = st.number_input("Dishwasher hours", 0.5, 6.0, 1.5, 0.5)
    dw_win = st.text_area(
        "Dishwasher windows",
        "11-10-2025 18:00:00, 11-10-2025 23:00:00"
    )

with col2:
    st.markdown("**Clothes dryer**")
    dr_on  = st.checkbox("Include Dryer", True)
    dr_p   = st.number_input("Dryer power (kW)", 0.0, 10.0, 4.5, 0.1)
    dr_h   = st.number_input("Dryer hours", 0.25, 4.0, 0.75, 0.25)
    dr_win = st.text_area(
        "Dryer windows",
        "11-13-2025 13:00:00, 11-13-2025 19:00:00"
    )

    st.markdown("---")
    st.markdown("**Washing machine**")
    wa_on  = st.checkbox("Include Washer", True)
    wa_p   = st.number_input("Washer power (kW)", 0.0, 5.0, 0.5, 0.1)
    wa_h   = st.number_input("Washer hours", 0.25, 4.0, 1.0, 0.25)
    wa_win = st.text_area(
        "Washer windows",
        "11-13-2025 12:00:00, 11-13-2025 17:00:00"
    )

st.info("Tip: Enter each availability window on its own line as:  `MM-DD-YYYY HH:MM:SS, MM-DD-YYYY HH:MM:SS`")

if st.button("Optimize", type="primary"):
    # Use LOCAL widget values directly (no session_state keys)
    START = datetime.combine(start_date, start_time)
    END   = datetime.combine(end_date, end_time)

    devices = []
    if ev_on: devices.append({"name":"EV", "P_kw":ev_p, "hours":ev_h, "windows":_parse_window_lines(ev_win)})
    if dr_on: devices.append({"name":"Dryer", "P_kw":dr_p, "hours":dr_h, "windows":_parse_window_lines(dr_win)})
    if dw_on: devices.append({"name":"Dishwasher", "P_kw":dw_p, "hours":dw_h, "windows":_parse_window_lines(dw_win)})
    if wa_on: devices.append({"name":"WashingMachine", "P_kw":wa_p, "hours":wa_h, "windows":_parse_window_lines(wa_win)})

    # Basic checks
    if not Path(data_file).exists():
        st.error(f"Data file not found: {data_file}")
    elif START >= END:
        st.error("End time must be after Start time.")
    elif not devices:
        st.error("Please include at least one device.")
    else:
        with st.spinner("Running optimization..."):
            results_df, schedule_df, pngs, xlsx = run_optimization(
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
            st.error("Model infeasible or no data in the selected window.\nTry raising Grid max power, lowering device power/hours, or relaxing Final SOC tolerance.")
        else:
            st.success("Optimization complete.")

            st.subheader("Timeseries (preview)")
            st.dataframe(results_df.head(200))

            st.subheader("Schedule")
            st.dataframe(schedule_df)

            # Show plots if they exist
            for p in (pngs or []):
                if p and Path(p).exists():
                    st.image(str(p), use_column_width=True)

            # Download Excel if it exists
            if xlsx and Path(xlsx).exists():
                st.download_button(
                    "Download Excel (Timeseries + Schedule)",
                    data=open(xlsx, "rb").read(),
                    file_name=Path(xlsx).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
