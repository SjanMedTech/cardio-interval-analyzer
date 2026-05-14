import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

fs = 1000

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

    # ECG R peak detection
    r_peaks, _ = find_peaks(
        ecg,
        distance=int(0.45*fs),
        prominence=0.6*np.std(ecg)
    )

    # Q peak detection
    q_peaks = []
    for r in r_peaks:
        start = max(r - int(0.05*fs), 0)
        end = r - int(0.015*fs)

        if end > start:
            q_peaks.append(start + np.argmin(ecg[start:end]))
        else:
            q_peaks.append(np.nan)

    q_peaks = np.array(q_peaks)

    # SCG detection
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

            neg_peaks, props = find_peaks(
                -ac_win,
                prominence=0.2*np.std(ac_win)
            )

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

                pos_peaks, _ = find_peaks(
                    mo_win,
                    prominence=0.15*np.std(mo_win)
                )

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

    # Cardiac Time Intervals
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

    # Plot
    N = int(10*fs)
    t = np.arange(N)/fs

    fig, ax = plt.subplots(2,1,figsize=(12,7))

    ax[0].plot(t, ecg[:N])
    ax[0].scatter(R_sec[R_sec<10], ecg[R[R<N]], c='red')
    ax[0].scatter(Q_sec[Q_sec<10], ecg[Q[Q<N].astype(int)], c='black')
    ax[0].set_title(title + " ECG")

    ax[1].plot(t, scg[:N])

    for arr,col in [(MC_sec,'blue'),(AO_sec,'orange'),(AC_sec,'green'),(MO_sec,'purple')]:
        valid = arr[~np.isnan(arr)]
        valid = valid[valid < 10]
        ax[1].scatter(valid, scg[(valid*fs).astype(int)], c=col)

    ax[1].set_title(title + " SCG")

    plt.tight_layout()

    return table, fig


# ------------------------------------------------------------
# STREAMLIT GUI
# ------------------------------------------------------------

st.title("ECG-SCG Cardiac Time Interval Detection")

st.write("Upload REST and POST exercise datasets")

rest_file = st.file_uploader("Upload REST Excel", type=["xlsx"])
post_file = st.file_uploader("Upload POST Excel", type=["xlsx"])

if rest_file and post_file:

    rest_df = pd.read_excel(rest_file)
    post_df = pd.read_excel(post_file)

    st.subheader("REST SIGNAL ANALYSIS")

    rest_table, rest_fig = detect_and_plot(rest_df,"REST")

    st.pyplot(rest_fig)
    st.dataframe(rest_table)

    st.subheader("POST EXERCISE SIGNAL ANALYSIS")

    post_table, post_fig = detect_and_plot(post_df,"POST")

    st.pyplot(post_fig)
    st.dataframe(post_table)

    final = pd.concat([rest_table, post_table], ignore_index=True)

    final.to_excel("CTI_RESULTS.xlsx", index=False)

    with open("CTI_RESULTS.xlsx","rb") as f:
        st.download_button("Download Results Excel",f,"CTI_RESULTS.xlsx")
