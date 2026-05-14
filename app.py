import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import io

fs = 1000

# ------------------------------------------------------------
# 🎨 CUSTOM CSS
# ------------------------------------------------------------
st.markdown("""
<style>
body {background-color: #F4F8FB;}
.metric-card {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin-bottom:10px;
}
.rest {background-color: #FFE5E5;}
.post {background-color: #E3F2FD;}
.section-title {
    font-size:22px;
    font-weight:600;
    margin-top:20px;
    color:#1D3557;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# ❤️ HEADER
# ------------------------------------------------------------
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
<img src="https://i.gifer.com/7efs.gif" width="250">
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🧭 NAVIGATION
# ------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Analysis", "📈 Comparison", "ℹ️ About"]
)

# Sidebar info
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
                depth = -ac_win[neg_peaks]
                prom = props["prominences"]
                score = depth*0.7 + prom*0.3
                best_idx = neg_peaks[np.argmax(score)]
                ac = r + (ac_start + best_idx)

        mo = np.nan
        if not np.isnan(ac):
            mo_start = int(ac - r + 0.02*fs)
            mo_end = int(ac - r + 0.12*fs)
            mo_win = beat[mo_start:mo_end]

            pos_peaks, _ = find_peaks(mo_win, prominence=0.15*np.std(mo_win))
            if len(pos_peaks) > 0:
                mo = r + mo_start + pos_peaks[np.argmax(mo_win[pos_peaks])]

        MC.append(mc); AO.append(ao); AC.append(ac); MO.append(mo)

    min_len = min(len(MC), len(AO), len(AC), len(MO), len(q_peaks))
    Q = q_peaks[:min_len]
    MC, AO, AC, MO = map(lambda x: np.array(x[:min_len]), [MC, AO, AC, MO])

    PEP = AO/fs - Q/fs
    LVET = AC/fs - AO/fs
    IVCT = AO/fs - MC/fs
    IVRT = MO/fs - AC/fs

    table = pd.DataFrame({
        "PEP_sec": PEP,
        "LVET_sec": LVET,
        "IVCT_sec": IVCT,
        "IVRT_sec": IVRT
    })

    fig, ax = plt.subplots(2,1,figsize=(10,6))
    ax[0].plot(ecg[:10000]); ax[0].set_title(title+" ECG")
    ax[1].plot(scg[:10000]); ax[1].set_title(title+" SCG")

    return table, fig, HR

# ------------------------------------------------------------
# 🏠 HOME
# ------------------------------------------------------------
if page == "🏠 Home":
    st.markdown("## Welcome")
    st.write("Upload ECG & SCG signals to analyze cardiac function.")

# ------------------------------------------------------------
# 📊 ANALYSIS
# ------------------------------------------------------------
elif page == "📊 Analysis":

    rest_file = st.file_uploader("Upload REST Excel", type=["xlsx"])
    post_file = st.file_uploader("Upload POST Excel", type=["xlsx"])

    if rest_file and post_file:

        rest_df = pd.read_excel(rest_file)
        post_df = pd.read_excel(post_file)

        with st.spinner("Processing..."):

            st.markdown('<div class="section-title">REST ANALYSIS</div>', unsafe_allow_html=True)
            rest_table, rest_fig, rest_hr = detect_and_plot(rest_df, "REST")

            st.markdown(f'<div class="metric-card rest"><h3>REST HR</h3><h2>{round(rest_hr,2)} bpm</h2></div>', unsafe_allow_html=True)
            st.pyplot(rest_fig)
            st.dataframe(rest_table)

            st.markdown('<div class="section-title">POST ANALYSIS</div>', unsafe_allow_html=True)
            post_table, post_fig, post_hr = detect_and_plot(post_df, "POST")

            st.markdown(f'<div class="metric-card post"><h3>POST HR</h3><h2>{round(post_hr,2)} bpm</h2></div>', unsafe_allow_html=True)
            st.pyplot(post_fig)
            st.dataframe(post_table)

            # SAVE FOR COMPARISON
            st.session_state.rest_table = rest_table
            st.session_state.post_table = post_table

# ------------------------------------------------------------
# 📈 COMPARISON
# ------------------------------------------------------------
elif page == "📈 Comparison":

    if "rest_table" in st.session_state:

        rest_mean = st.session_state.rest_table.mean()
        post_mean = st.session_state.post_table.mean()

        fig, ax = plt.subplots()
        x = np.arange(len(rest_mean))

        ax.bar(x-0.2, rest_mean, 0.4, label="REST")
        ax.bar(x+0.2, post_mean, 0.4, label="POST")

        ax.set_xticks(x)
        ax.set_xticklabels(rest_mean.index)
        ax.legend()

        st.pyplot(fig)

    else:
        st.warning("Run Analysis first")

# ------------------------------------------------------------
# ℹ️ ABOUT
# ------------------------------------------------------------
elif page == "ℹ️ About":
    st.markdown("""
    ## About

    Biomedical application to estimate cardiac intervals using ECG + SCG.

    Useful for:
    - Cardiac dysfunction screening
    - Wearable diagnostics
    """)
