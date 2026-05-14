import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import io
import time

fs = 1000

# ------------------------------------------------------------
# 🎨 CUSTOM CSS
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
.section-title {
    font-size:22px;
    font-weight:600;
    margin-top:20px;
    color:#1D3557;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🧭 NAVIGATION
# ------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Analysis", "📈 Comparison", "ℹ️ About"]
)

# ------------------------------------------------------------
# ❤️ HEADER (ONLY ON HOME)
# ------------------------------------------------------------
if page == "🏠 Home":

    st.markdown("""
    <h1 style='text-align: center; color: #E63946;'>
    ❤️ Cardiac Time Interval Analyzer
    </h1>
    <p style='text-align: center; color: #1D3557; font-size:18px;'>
    ECG + SCG based Cardiac Dysfunction Assessment
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;">
    <img src="https://i.gifer.com/7efs.gif" width="300">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Welcome 👋")
    st.write("This app analyzes ECG + SCG signals to estimate cardiac time intervals.")

    st.markdown("""
    ### 🧪 Workflow
    ECG + SCG → Filtering → Peak Detection → CTI Estimation → Visualization
    """)

# ------------------------------------------------------------
# SIDEBAR INFO
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
# DETECTION FUNCTION
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

    MC, AO, AC, MO = [], [], [], []

    for i in range(len(r_peaks)-1):
        r = r_peaks[i]
        next_r = r_peaks[i+1]
        beat = scg[r:next_r]

        if len(beat) < int(0.3*fs):
            continue

        mc = r + np.argmin(beat[0:int(0.06*fs)])

        ao_win = beat[int(0.04*fs):int(0.18*fs)]
        pos_peaks, _ = find_peaks(ao_win, prominence=0.25*np.std(ao_win))
        if len(pos_peaks) == 0:
            continue

        ao_rel = int(0.04*fs) + pos_peaks[np.argmax(ao_win[pos_peaks])]
        ao = r + ao_rel

        ac = np.nan
        ac_rel = None

        ac_start = ao_rel + int(0.05*fs)
        ac_end = min(int(0.45*fs), len(beat))

        if ac_start < ac_end:
            ac_win = beat[ac_start:ac_end]
            neg_peaks, props = find_peaks(-ac_win, prominence=0.2*np.std(ac_win))

            if len(neg_peaks) > 0:
                best_idx = neg_peaks[np.argmax(-ac_win[neg_peaks])]
                ac_rel = ac_start + best_idx
                ac = r + ac_rel

        mo = np.nan
        if ac_rel is not None:
            mo_start = ac_rel + int(0.02*fs)
            mo_end = min(ac_rel + int(0.12*fs), len(beat))

            if mo_start < mo_end:
                mo_win = beat[mo_start:mo_end]
                pos_peaks, _ = find_peaks(mo_win)

                if len(pos_peaks) > 0:
                    best_idx = pos_peaks[np.argmax(mo_win[pos_peaks])]
                    mo_rel = mo_start + best_idx
                    mo = r + mo_rel

        MC.append(mc)
        AO.append(ao)
        AC.append(ac)
        MO.append(mo)

    min_len = min(len(MC), len(AO), len(AC), len(MO))

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
# 📊 ANALYSIS PAGE
# ------------------------------------------------------------
elif page == "📊 Analysis":

    st.markdown("## 📊 Signal Analysis")

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

            st.markdown(f'<div class="metric-card rest"><h3>REST HR</h3><h2>{round(rest_hr,2)} bpm</h2></div>', unsafe_allow_html=True)

            if rest_hr > 100:
                st.error("⚠️ Elevated Heart Rate")
            else:
                st.success("✅ Normal Range")

            st.pyplot(rest_fig)
            st.dataframe(rest_table)

        with tab2:
            post_table, post_fig, post_hr = detect_and_plot(post_df, "POST")

            st.markdown(f'<div class="metric-card post"><h3>POST HR</h3><h2>{round(post_hr,2)} bpm</h2></div>', unsafe_allow_html=True)

            if post_hr > 120:
                st.error("⚠️ High Cardiac Load")
            else:
                st.success("✅ Expected Response")

            st.pyplot(post_fig)
            st.dataframe(post_table)

        # SAVE FOR NEXT PAGE
        st.session_state.rest_table = rest_table
        st.session_state.post_table = post_table

# ------------------------------------------------------------
# 📈 COMPARISON PAGE
# ------------------------------------------------------------
elif page == "📈 Comparison":

    st.markdown("## 📈 REST vs POST Comparison")

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
        st.warning("Run Analysis first")

# ------------------------------------------------------------
# ℹ️ ABOUT PAGE
# ------------------------------------------------------------
elif page == "ℹ️ About":

    st.markdown("""
    ## About This Project

    This application estimates cardiac time intervals using ECG and SCG signals.

    ### Features:
    - Signal filtering
    - Peak detection
    - CTI estimation
    - REST vs POST comparison

    ### Use Case:
    Early detection of cardiac dysfunction using wearable sensors.
    """)
