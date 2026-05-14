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
body {
    background-color: #F4F8FB;
}
.metric-card {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 10px;
}
.rest {
    background-color: #FFE5E5;
}
.post {
    background-color: #E3F2FD;
}
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

# ------------------------------------------------------------
# 🚀 WELCOME SCREEN (POP-UP STYLE)
# ------------------------------------------------------------
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:

    st.markdown("""
    <div style="
        text-align:center;
        padding:50px;
        background: linear-gradient(135deg, #FFE5E5, #E3F2FD);
        border-radius:15px;
        box-shadow:0px 4px 20px rgba(0,0,0,0.1);
    ">
        <h1 style="color:#E63946;">🫀 Cardiac Time Interval Analyzer</h1>
        <p style="font-size:18px; color:#1D3557;">
        Welcome
        </p>
        <p style="font-size:16px;">
        Analyze cardiac function using ECG + SCG signals  
        and estimate cardiac time intervals.
        </p>
        <img src="https://i.gifer.com/7efs.gif" width="200">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Enter Application"):
        st.session_state.started = True

    st.stop()

# ------------------------------------------------------------
# SIDEBAR
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

        MC.append(mc)
        AO.append(ao)

    min_len = min(len(MC), len(AO), len(r_peaks), len(q_peaks))

    R = r_peaks[:min_len]
    Q = q_peaks[:min_len]
    MC = np.array(MC[:min_len])
    AO = np.array(AO[:min_len])

    R_sec = R / fs
    Q_sec = Q / fs
    MC_sec = MC / fs
    AO_sec = AO / fs

    PEP = AO_sec - Q_sec
    IVCT = AO_sec - MC_sec

    table = pd.DataFrame({
        "PEP_sec": PEP,
        "IVCT_sec": IVCT
    })

    fig, ax = plt.subplots(2,1,figsize=(10,6))

    ax[0].plot(ecg[:10000])
    ax[0].set_title(title + " ECG")

    ax[1].plot(scg[:10000])
    ax[1].set_title(title + " SCG")

    plt.tight_layout()

    return table, fig, HR

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
rest_file = st.file_uploader("Upload REST Excel", type=["xlsx"])
post_file = st.file_uploader("Upload POST Excel", type=["xlsx"])

if rest_file and post_file:

    rest_df = pd.read_excel(rest_file)
    post_df = pd.read_excel(post_file)

    with st.spinner("Processing..."):

        # REST
        st.markdown('<div class="section-title">REST ANALYSIS</div>', unsafe_allow_html=True)
        rest_table, rest_fig, rest_hr = detect_and_plot(rest_df, "REST")

        st.markdown(f'<div class="metric-card rest"><h3>REST HR</h3><h2>{round(rest_hr,2)} bpm</h2></div>', unsafe_allow_html=True)
        st.pyplot(rest_fig)
        st.dataframe(rest_table)

        # POST
        st.markdown('<div class="section-title">POST ANALYSIS</div>', unsafe_allow_html=True)
        post_table, post_fig, post_hr = detect_and_plot(post_df, "POST")

        st.markdown(f'<div class="metric-card post"><h3>POST HR</h3><h2>{round(post_hr,2)} bpm</h2></div>', unsafe_allow_html=True)
        st.pyplot(post_fig)
        st.dataframe(post_table)

        # COMPARISON
        st.markdown('<div class="section-title">📊 COMPARISON</div>', unsafe_allow_html=True)

        rest_mean = rest_table.mean()
        post_mean = post_table.mean()

        fig2, ax2 = plt.subplots()
        x = np.arange(len(rest_mean))

        ax2.bar(x - 0.2, rest_mean, 0.4, label="REST")
        ax2.bar(x + 0.2, post_mean, 0.4, label="POST")

        ax2.set_xticks(x)
        ax2.set_xticklabels(rest_mean.index)
        ax2.legend()

        st.pyplot(fig2)

        # DOWNLOAD
        output = io.BytesIO()
        pd.concat([rest_table, post_table]).to_excel(output, index=False)

        st.download_button("Download Results", output.getvalue(), "CTI_RESULTS.xlsx")
