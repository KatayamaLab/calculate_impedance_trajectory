import os
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

@st.cache_data
def plot_graph_1(real, imag, folder_path, key):

    fig, ax = plt.subplots()
    ax.scatter(real, imag, label='Measured Data')

    analyzer_data_path = pathlib.Path(os.path.join(folder_path, 'インピーダンスアナライザ'))
    analyzer_freq_list = list(analyzer_data_path.glob("*.csv"))
    for f in analyzer_freq_list:
        analyzer_df = pd.read_csv(f, skiprows=18)
        analyzer_ch1 = analyzer_df.iloc[0:21, 2]
        analyzer_ch2 = -1 * analyzer_df.iloc[0:21, 3]
        ax.scatter(analyzer_ch1, analyzer_ch2, label='Analyzer Data')

    ax.set_xlabel("Z' / Ω")
    ax.set_ylabel('Z" / Ω')
    ax.set(aspect=1)
    # ax.set_title('Impedance Analysis')
    ax.legend()
    ax.grid(which='major',color='grey',linestyle='--')

    if st.session_state['tick_nyquist'] is not None:
        state = st.session_state['tick_nyquist']
        ax.set_xlim(*state['x'])
        ax.set_ylim(*state['y'])
    elif st.session_state['tick_nyquist'] is None:
        st.session_state['tick_nyquist'] = {
            'x': ax.get_xlim(),
            'y': ax.get_ylim()
        }

    return fig

@st.cache_data
def FFT_graph(ch1_spectre, ch2_spectre, frequency, idx_ch1, idx_ch2, key):
    fig = plt.figure(figsize=(20, 10))

    ax_ch1 = fig.add_subplot(121)
    ax_ch1.set_xlim((1, 10**4))
    ax_ch1.set_xscale("log")
    ax_ch1.set_ylabel('current[A]', fontsize=18)
    ax_ch1.set_xlabel('frequency[Hz]', fontsize=18)
    ax_ch1.set_xticklabels(ax_ch1.get_xticklabels(),fontsize=18, rotation=45)
    ax_ch1.set_yticklabels(ax_ch1.get_yticklabels(),fontsize=18)

    for k, v in ch1_spectre.items():
        ax_ch1.plot([k, k], [0, 0.05], c=cm.jet((k - 10) / (1000 - 10)), linestyle="dashed")
        ax_ch1.plot(frequency[k], v, label=f"{k} Hz", c=cm.jet((k - 10) / (1000 - 10)))
        ax_ch1.scatter([frequency[k][idx_ch1[k]]], v[idx_ch1[k]], marker="*", color=cm.jet((k - 10) / (1000 - 10)))

    ax_ch2 = fig.add_subplot(122)
    ax_ch2.set_xlim((1, 10**4))
    ax_ch2.set_xscale("log")
    ax_ch2.set_ylabel('Voltage[V]', fontsize=18)
    ax_ch2.set_xlabel('frequency[Hz]', fontsize=18)
    ax_ch2.set_xticklabels(ax_ch2.get_xticklabels(),fontsize=18, rotation=45)
    ax_ch2.set_yticklabels(ax_ch2.get_yticklabels(),fontsize=18)


    for k, v in ch2_spectre.items():
        ax_ch2.plot([k, k], [0, 0.05], c=cm.jet((k - 10) / (1000 - 10)), linestyle="dashed")
        ax_ch2.plot(frequency[k], v, label=f"{k} Hz", c=cm.jet((k - 10) / (1000 - 10)))
        ax_ch2.scatter([frequency[k][idx_ch2[k]]], v[idx_ch2[k]], marker="*", color=cm.jet((k - 10) / (1000 - 10)))

    ax_ch1.legend()


    if st.session_state['tick_spectre'] is not None:
        state = st.session_state['tick_spectre']
        ax_ch1.set_ylim(*state['y1'])
        ax_ch2.set_ylim(*state['y2'])
    elif st.session_state['tick_spectre'] is None:
        st.session_state['tick_spectre'] = {
            'y1': ax_ch1.get_ylim(),
            'y2': ax_ch2.get_ylim()
        }

    return fig

@st.cache_data
def Bode_plot(real, imag, frequency, folder_path):
    analyzer_data_path = pathlib.Path(os.path.join(folder_path, 'インピーダンスアナライザ'))
    analyzer_freq_list = list(analyzer_data_path.glob("*.csv"))
    for f in analyzer_freq_list:
        analyzer_df = pd.read_csv(f, skiprows=18)
        analyzer_ch1 = analyzer_df.iloc[0:21, 2]
        analyzer_ch2 = -1 * analyzer_df.iloc[0:21, 3]
    analyzer_impedance = [complex(r, i) for r, i in zip(analyzer_ch1, analyzer_ch2)]
        
    
    impedance = [complex(r, i) for r, i in zip(real, imag)]
    impedance_analyzer = [complex(r, i) for r, i in zip(analyzer_ch1, analyzer_ch2)]
    
    gain = np.abs(impedance)
    gain_analyzer = np.abs(impedance_analyzer)
    theta = np.angle(impedance, deg= True)
    theta_analyzer = np.angle(impedance_analyzer, deg= True)

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(211)
    # ax1.set_xlim((1, 10**4))
    ax1.set_ylim(0.0, 0.1)
    ax1.set_xscale("log")
    ax1.set_ylabel('|Z| [Ω]', fontsize=18)

    ax1.scatter(frequency, gain, label='Measured Data')
    ax1.scatter(frequency, gain_analyzer, label='Analyzer Data')
    ax1.grid(which='major',color='grey',linestyle='--')
    ax1.grid(which='minor',color='gray',linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.legend(fontsize=18)

    ax2 = fig.add_subplot(212, sharex=ax1)
    # ax2.set_xlim((1, 10**4))
    ax2.set_ylim(-40, 40)
    ax2.set_xscale("log")
    ax2.set_ylabel('θ [°]', fontsize=18)
    ax2.set_xlabel('Frequency [Hz]', fontsize=18)
    ax2.scatter(frequency, theta)
    ax2.scatter(frequency, theta_analyzer)
    ax2.grid(which='major',color='grey',linestyle='--')
    ax2.grid(which='minor',color='gray',linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=18)
    return fig



