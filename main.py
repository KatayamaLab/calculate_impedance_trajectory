import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from natsort import natsorted
from scipy.signal import windows
import matplotlib.cm as cm
import streamlit as st
import os

st.title('解析アプリ')

shunt_registor = st.selectbox(
    'シャント抵抗を選択してください(Ω)',
    [1, 0.1, 0.01],
    index=None,
    placeholder="選択してください"
)

vector = st.selectbox(
    '電流の向きを選んでください',
    [1.0, -1.0],
    index=None,
    placeholder="選択してください"
)

if st.button('キャッシュをクリア'):
    st.cache_data.clear()
    st.success('キャッシュをクリアしました。')

base_dir = '/Users/ken/Library/CloudStorage/Box-Box/片山研究室・共有フォルダ/1_研究データ/1_2024_LIB_EIS_ML/4_パルス幅を変化させてEIS'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

folder_name = st.text_input('作成するフォルダの名前を入力してください:')

if st.button('フォルダを作成'):
    if folder_name:
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            st.success(f"フォルダ '{folder_name}' が作成されました。")
        else:
            st.warning(f"フォルダ '{folder_name}' は既に存在します。")
    else:
        st.error('フォルダ名を入力してください。')

if folder_name:
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(os.path.join(folder_path, 'result'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'インピーダンスアナライザ'), exist_ok=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("オシロスコープのデータをアップロード", accept_multiple_files=True)
with col2:
    analyzer_uploaded_files = st.file_uploader("インピーダンスアナライザのデータをアップロード", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.session_state['uploaded_files'].append(uploaded_file)
        file_path = os.path.join(folder_path, 'result', uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

if analyzer_uploaded_files:
    for analyzer_uploaded_file in analyzer_uploaded_files:
        st.session_state['uploaded_files'].append(analyzer_uploaded_file)
        file_path = os.path.join(folder_path, 'インピーダンスアナライザ', analyzer_uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(analyzer_uploaded_file.getbuffer())

folder_selection = st.selectbox(
    '解析するフォルダを選択してください:',
    natsorted([f.name for f in pathlib.Path(base_dir).iterdir() if f.is_dir()]),
    index=0,
    placeholder="フォルダを選択してください"
)
folder_path = os.path.join(base_dir, folder_selection)
st.write(f"選択したフォルダは{folder_selection}です")

average_options = [f'average{i}' for i in range(1, 11)]
average_selection = st.multiselect(
    '解析する平均値を選択してください:',
    average_options,
    placeholder="平均値を選択してください"
)


@st.cache_data
def make_graph(selected_averages):
    data_path = pathlib.Path(os.path.join(folder_path, 'result'))
    freq_list = list(
        set(
            map(
                lambda p: int(str(p.stem).split("-")[0].replace("frequency", "")),
                data_path.glob("*.CSV"),
            )
        )
    )
    real, imag = [], []
    idx_ch1, idx_ch2 = {}, {}
    frequency, ch1_spectre, ch2_spectre = {}, {}, {}

    for f in natsorted(freq_list):
        impedance_list = []
        for avg in selected_averages:
            pattern = f"frequency{f}-{avg}.CSV"
            for p in data_path.glob(pattern):
                ch1_buff, ch2_buff = [], []
                st.write(p)
                Ts = float(pd.read_csv(p, nrows=20).loc["Sampling Period"].iat[0, 1])
                meas_data = pd.read_csv(p, skiprows=25)
                window_f = windows.blackman(len(meas_data))
                meas_data.iloc[:, 1] = meas_data.iloc[:, 1] * window_f
                meas_data.iloc[:, 3] = meas_data.iloc[:, 3] * (1/shunt_registor) * window_f * vector
                freq = np.fft.rfftfreq(len(meas_data), d=Ts)[1:]
                F_ch1 = np.fft.rfft(meas_data.iloc[:, 1].to_numpy())[1:]
                F_ch2 = np.fft.rfft(meas_data.iloc[:, 3].to_numpy())[1:]
                freq_lower = f * 0.85
                freq_upper = f * 1.15
                F_ch1_ = F_ch1.copy()
                F_ch2_ = F_ch2.copy()
                F_ch1_[~((freq_lower < freq) & (freq < freq_upper))] = 0
                F_ch2_[~((freq_lower < freq) & (freq < freq_upper))] = 0
                max_spectre_idx_v = np.argmax(np.abs(F_ch1_))
                max_spectre_idx_i = np.argmax(np.abs(F_ch2_))
                idx_ch1[f] = max_spectre_idx_v
                idx_ch2[f] = max_spectre_idx_i
                N = len(meas_data)
                impedance_list.append(F_ch1[max_spectre_idx_v] / F_ch2[max_spectre_idx_i])
                ch1_buff.append(np.abs(F_ch1) / (N / 2))
                ch2_buff.append(np.abs(F_ch2) / (N / 2))
        impedance = np.average(impedance_list, axis=0)
        real.append(impedance.real)
        imag.append(-impedance.imag)
        frequency[f] = freq
        ch1_spectre[f] = np.average(ch1_buff, axis=0)
        ch2_spectre[f] = np.average(ch2_buff, axis=0)
    
    return real, imag, ch1_spectre, ch2_spectre, frequency, idx_ch1, idx_ch2

def plot_graph_1(real, imag, x_min, x_max, y_min, y_max, plot_container):
    fig, ax = plt.subplots()
    ax.scatter(real, imag, label='Measured Data')

    analyzer_data_path = pathlib.Path(os.path.join(folder_path, 'インピーダンスアナライザ'))
    analyzer_freq_list = list(analyzer_data_path.glob("*.csv"))
    for f in analyzer_freq_list:
        analyzer_df = pd.read_csv(f, skiprows=18)
        analyzer_ch1 = analyzer_df.iloc[0:21, 2]
        analyzer_ch2 = -1 * analyzer_df.iloc[0:21, 3]
        ax.scatter(analyzer_ch1, analyzer_ch2, label='Analyzer Data')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel('Real Part of Impedance')
    ax.set_ylabel('Imaginary Part of Impedance')
    ax.set_title('Impedance Analysis')
    ax.legend()
    plot_container.pyplot(fig)

def FFT_graph(ch1_spectre, ch2_spectre, frequency, idx_ch1, idx_ch2, y1_max, y2_max, plot_container):
    fig = plt.figure(figsize=(20, 10))

    ax_ch1 = fig.add_subplot(121)
    ax_ch1.set_xlim((1, 10**4))
    ax_ch1.set_ylim(0, y1_max)
    ax_ch1.set_xscale("log")
    ax_ch1.set_ylabel('current[A]')
    ax_ch1.set_xlabel('frequency[Hz]')

    for k, v in ch1_spectre.items():
        ax_ch1.plot([k, k], [0, 0.05], c=cm.jet((k - 10) / (1000 - 10)), linestyle="dashed")
        ax_ch1.plot(frequency[k], v, label=f"{k} Hz", c=cm.jet((k - 10) / (1000 - 10)))
        ax_ch1.scatter([frequency[k][idx_ch1[k]]], v[idx_ch1[k]], marker="*", color=cm.jet((k - 10) / (1000 - 10)))

    ax_ch2 = fig.add_subplot(122)
    ax_ch2.set_xlim((1, 10**4))
    ax_ch2.set_ylim(0, y2_max)
    ax_ch2.set_xscale("log")
    ax_ch2.set_ylabel('Voltage[V]')
    ax_ch2.set_xlabel('frequency[Hz]')

    for k, v in ch2_spectre.items():
        ax_ch2.plot([k, k], [0, 0.05], c=cm.jet((k - 10) / (1000 - 10)), linestyle="dashed")
        ax_ch2.plot(frequency[k], v, label=f"{k} Hz", c=cm.jet((k - 10) / (1000 - 10)))
        ax_ch2.scatter([frequency[k][idx_ch2[k]]], v[idx_ch2[k]], marker="*", color=cm.jet((k - 10) / (1000 - 10)))

    ax_ch1.legend()
    plot_container.pyplot(fig)


if not average_selection:
    st.error('平均値を選択してください。')

else:
    
    x_min = st.sidebar.number_input('X軸の最小値を入力してください', value=0.05)
    x_max = st.sidebar.number_input('X軸の最大値を入力してください', value=0.1)
    y_min = st.sidebar.number_input('Y軸の最小値を入力してください', value=-0.005)
    y_max = st.sidebar.number_input('Y軸の最大値を入力してください', value=0.01)
    y1_max = st.sidebar.number_input('Ch1のY軸の最大値を入力してください', value=0.03)
    y2_max = st.sidebar.number_input('Ch2のY軸の最大値を入力してください', value=0.03)
    x_length = st.sidebar.number_input('実軸の補正値を入力してください', value=0.0)
    real, imag, ch1_spectre, ch2_spectre, frequency, idx_ch1, idx_ch2 = make_graph(average_selection)
    real = [x_length + i for i in real]
    plot_container = st.container()
    if st.button('グラフ作成'):
        plot_graph_1(real, imag, x_min, x_max, y_min, y_max, plot_container)
        FFT_graph(ch1_spectre, ch2_spectre, frequency, idx_ch1, idx_ch2, y1_max, y2_max, plot_container)
        st.success('グラフ作成が完了しました。')