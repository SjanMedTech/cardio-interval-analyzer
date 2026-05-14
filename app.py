import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import io

fs = 1000

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ---------------- CSS ----------------
st.markdown("""
<style>
.metric-card {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
.rest { background-color: #FFE5E5; }
.post { background-color: #E3F2FD; }
</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ----------------
def bandpass(x, low, high):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def detect(df):

    ecg = bandpass(df["II"].values, 5, 25)
    scg = bandpass(df["az"].values, 7, 30)

    r_peaks, _ = find_peaks(ecg, distance=int(0.45*fs), prominence=0.6*np.std(ecg))

    R_sec = r_peaks / fs
    HR = 60 / np.mean(np.diff(R_sec)) if len(R_sec) > 1 else np.nan

    return ecg, scg, HR

# ---------------- PAGE 1: WELCOME ----------------
if st.session_state.page == "welcome":

    st.markdown("""
    <div style="text-align:center;padding:50px;
    background: linear-gradient(135deg, #FFE5E5, #E3F2FD);
    border-radius:15px;">
        <h1 style="color:#E63946;">🫀 Cardiac Time Interval Analyzer</h1>
        <p>Welcome</p>
        <img src="https://i.gifer.com/7efs.gif" width="200">
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Start"):
        st.session_state.page = "upload"
        st.rerun()

# ---------------- PAGE 2: UPLOAD ----------------
elif st.session_state.page == "upload":

    st.title("Upload Data")

    rest_file = st.file_uploader("Upload REST Excel", type=["xlsx"])
    post_file = st.file_uploader("Upload POST Excel", type=["xlsx"])

    if rest_file and post_file:
        st.session_state.rest_df = pd.read_excel(rest_file)
        st.session_state.post_df = pd.read_excel(post_file)

        if st.button("➡️ Process Data"):
            st.session_state.page = "results"
            st.rerun()

# ---------------- PAGE 3: RESULTS ----------------
elif st.session_state.page == "results":

    st.title("📊 Results")

    rest_df = st.session_state.rest_df
    post_df = st.session_state.post_df

    ecg_r, scg_r, rest_hr = detect(rest_df)
    ecg_p, scg_p, post_hr = detect(post_df)

    # REST
    st.subheader("REST")
    st.markdown(f'<div class="metric-card rest"><h3>{round(rest_hr,2)} bpm</h3></div>', unsafe_allow_html=True)

    fig1, ax1 = plt.subplots()
    ax1.plot(ecg_r[:5000])
    ax1.set_title("REST ECG")
    st.pyplot(fig1)

    # POST
    st.subheader("POST")
    st.markdown(f'<div class="metric-card post"><h3>{round(post_hr,2)} bpm</h3></div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots()
    ax2.plot(ecg_p[:5000])
    ax2.set_title("POST ECG")
    st.pyplot(fig2)

    # Comparison
    st.subheader("Comparison")

    fig3, ax3 = plt.subplots()
    ax3.bar(["REST","POST"], [rest_hr, post_hr])
    st.pyplot(fig3)

    # Back button
    if st.button("⬅️ Back to Upload"):
        st.session_state.page = "upload"
        st.rerun()
