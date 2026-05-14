import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import io

fs = 1000

# ------------------------------------------------------------
# 🎨 CUSTOM CSS (DASHBOARD LOOK)
# ------------------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #F4F8FB;
}
.card {
    padding: 15px;
    border-radius: 12px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}
.metric-card {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
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

# ECG animation (clean + relevant)
st.markdown("""
<div style="text-align:center;">
<img src="https://i.gifer.com/7efs.gif" width="300">
</div>
""", unsafe_allow_html=True)

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
                ac_rel = ac_start + best_idx
                ac = r + ac_rel

        mo = np.nan
        if ac_rel is not None:
            mo_start = ac_rel + int(0.02*fs)
            mo_end = min(ac_rel + int(0.12*fs), len(beat))

            if mo_start < mo_end:
                mo_win = beat[mo_start:mo_end]
                pos_peaks, _ = find_peaks(mo_win, prominence=0.15*np.std(mo_win))

                if len(pos_peaks) > 0:
                    best_idx = pos_peaks[np.argmax(mo_win[pos_peaks])]
                    mo_rel = mo_start + best_idx
                    mo = r + mo_rel

        MC.append(mc)
        AO.append(ao)
        AC.append(ac)
        MO.append(mo)

    min_len = min(len(MC), len(AO), len(AC), len(MO), len(r_peaks), len(q_peaks))

    R = r_peaks[:min_len]
    Q = q_peaks[:min_len]
    MC = np.array(MC[:min_len])
    AO = np.array(AO[:min_len])
    AC = np.array(AC[:min_len])
    MO = np.array(MO[:min_len])

    R_sec = R / fs
    Q_sec = Q / fs
    MC_sec = MC / fs
    AO_sec = AO / fs
    AC_sec = AC / fs
    MO_sec = MO / fs

    PEP = AO_sec - Q_sec
    LVET = AC_sec - AO_sec
    IVCT = AO_sec - MC_sec
    IVRT = MO_sec - AC_sec

    table = pd.DataFrame({
        "PEP_sec": PEP,
        "LVET_sec": LVET,
        "IVCT_sec": IVCT,
        "IVRT_sec": IVRT
    })

    fig, ax = plt.subplots(2,1,figsize=(12,7))

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

        output = io.BytesIO()
        pd.concat([rest_table, post_table]).to_excel(output, index=False)

        st.download_button("Download Results", output.getvalue(), "CTI_RESULTS.xlsx")
