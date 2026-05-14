import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import io
import time

fs = 1000

# ------------------------------------------------------------
# 🎨 CSS
# ------------------------------------------------------------
st.markdown("""
<style>
.metric-card {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin-bottom:10px;
}
.rest { background-color: #FFE5E5; }
.post { background-color: #E3F2FD; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🧭 NAVIGATION (TOP LEVEL ONLY)
# ------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Analysis", "📈 Comparison", "ℹ️ About"]
)

# ------------------------------------------------------------
# 🫀 SIDEBAR INFO (SAFE OUTSIDE)
# ------------------------------------------------------------
st.sidebar.title("🫀 Cardiac Info")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3a/Heart_diagram-en.svg")
st.sidebar.write("""
**PEP** – Pre-Ejection Period  
**LVET** – Left Ventricular Ejection Time  
**IVCT** – Isovolumetric Contraction Time  
**IVRT** – Isovolumetric Relaxation Time  
""")

# ------------------------------------------------------------
# FILTER
# ------------------------------------------------------------
def bandpass(x, low, high):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)

# ------------------------------------------------------------
# DETECTION
# ------------------------------------------------------------
def detect_and_plot(df, title):

    ecg = bandpass(df["II"].values, 5, 25)
    scg = bandpass(df["az"].values, 7, 30)

    r_peaks, _ = find_peaks(ecg, distance=int(0.45*fs), prominence=0.6*np.std(ecg))

    R_sec = r_peaks / fs
    HR = 60 / np.mean(np.diff(R_sec)) if len(R_sec) > 1 else np.nan

    q_peaks = []
    for r in r_peaks:
        start = max(r - int(0.05*fs), 0)
        end = r - int(0.015*fs)
        q_peaks.append(start + np.argmin(ecg[start:end]) if end > start else np.nan)

    q_peaks = np.array(q_peaks)

    AO = []
    AC = []

    for i in range(len(r_peaks)-1):
        r = r_peaks[i]
        next_r = r_peaks[i+1]
        beat = scg[r:next_r]

        if len(beat) < int(0.3*fs):
            continue

        ao = r + np.argmax(beat[int(0.04*fs):int(0.18*fs)]) + int(0.04*fs)
        ac = r + np.argmin(beat[int(0.2*fs):int(0.45*fs)]) + int(0.2*fs)

        AO.append(ao)
        AC.append(ac)

    min_len = min(len(AO), len(AC), len(q_peaks))

    PEP = (np.array(AO[:min_len]) - np.array(q_peaks[:min_len])) / fs
    LVET = (np.array(AC[:min_len]) - np.array(AO[:min_len])) / fs

    table = pd.DataFrame({
        "PEP_sec": PEP,
        "LVET_sec": LVET
    })

    fig, ax = plt.subplots()
    ax.plot(ecg[:10000])
    ax.set_title(title)

    return table, fig, HR

# ------------------------------------------------------------
# 🏠 HOME
# ------------------------------------------------------------
if page == "🏠 Home":

    st.markdown("""
    <h1 style='text-align: center; color: #E63946;'>
    ❤️ Cardiac Time Interval Analyzer
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;">
    <img src="https://i.gifer.com/7efs.gif" width="300">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Welcome 👋")
    st.write("Analyze ECG + SCG signals to estimate cardiac intervals.")

# ------------------------------------------------------------
# 📊 ANALYSIS
# ------------------------------------------------------------
elif page == "📊 Analysis":

    st.header("📊 Analysis")

    rest_file = st.file_uploader("Upload REST Excel", type=["xlsx"])
    post_file = st.file_uploader("Upload POST Excel", type=["xlsx"])

    if rest_file and post_file:

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

        rest_df = pd.read_excel(rest_file)
        post_df = pd.read_excel(post_file)

        tab1, tab2 = st.tabs(["REST", "POST"])

        with tab1:
            rest_table, rest_fig, rest_hr = detect_and_plot(rest_df, "REST")
            st.markdown(f'<div class="metric-card rest"><h3>REST HR</h3><h2>{round(rest_hr,2)}</h2></div>', unsafe_allow_html=True)
            st.pyplot(rest_fig)
            st.dataframe(rest_table)

        with tab2:
            post_table, post_fig, post_hr = detect_and_plot(post_df, "POST")
            st.markdown(f'<div class="metric-card post"><h3>POST HR</h3><h2>{round(post_hr,2)}</h2></div>', unsafe_allow_html=True)
            st.pyplot(post_fig)
            st.dataframe(post_table)

        # SAVE STATE
        st.session_state.rest_table = rest_table
        st.session_state.post_table = post_table

# ------------------------------------------------------------
# 📈 COMPARISON
# ------------------------------------------------------------
elif page == "📈 Comparison":

    st.header("📈 Comparison")

    if "rest_table" in st.session_state:

        rest_table = st.session_state.rest_table
        post_table = st.session_state.post_table

        rest_mean = rest_table.mean()
        post_mean = post_table.mean()

        st.bar_chart(pd.DataFrame({
            "REST": rest_mean,
            "POST": post_mean
        }))

    else:
        st.warning("Run analysis first")

# ------------------------------------------------------------
# ℹ️ ABOUT
# ------------------------------------------------------------
elif page == "ℹ️ About":

    st.header("ℹ️ About")

    st.write("""
    Biomedical dashboard for cardiac interval estimation using ECG + SCG.

    Features:
    - Signal processing
    - Peak detection
    - CTI estimation
    - Comparison dashboard
    """)
