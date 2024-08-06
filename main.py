import os
import pathlib


import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import windows
from natsort import natsorted

import graph_util

BASE_DIR = '/Users/ken/Library/CloudStorage/Box-Box/片山研究室・共有フォルダ/1_研究データ/1_2024_LIB_EIS_ML/4_パルス幅を変化させてEIS'

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
                # st.write(p)
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



def clear_cache():
    st.cache_data.clear()
    st.success('キャッシュをクリアしました。')


def update_key():
    st.session_state['key'] += 1


st.title('解析アプリ')

shunt_registor = st.selectbox(
    'シャント抵抗を選択してください(Ω)',
    [1, 0.1, 0.01],
    index=0,
    placeholder="選択してください"
)

vector = st.selectbox(
    '電流の向きを選んでください',
    [1.0, -1.0],
    index=0,
    placeholder="選択してください"
)

if st.button('キャッシュをクリア'):
    clear_cache()


if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

folder_name = st.text_input('作成するフォルダの名前を入力してください:')

if st.button('フォルダを作成'):
    if folder_name:
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            st.success(f"フォルダ '{folder_name}' が作成されました。")
        else:
            st.warning(f"フォルダ '{folder_name}' は既に存在します。")
    else:
        st.error('フォルダ名を入力してください。')

if folder_name:
    folder_path = os.path.join(BASE_DIR, folder_name)
    os.makedirs(os.path.join(folder_path, 'result'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'インピーダンスアナライザ'), exist_ok=True)

# st.set_option('deprecation.showPyplotGlobalUse', False)

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
    natsorted([f.name for f in pathlib.Path(BASE_DIR).iterdir() if f.is_dir()]),
    index=0,
    placeholder="フォルダを選択してください",
    on_change=clear_cache
)

folder_path = os.path.join(BASE_DIR, folder_selection)
st.write(f"選択したフォルダは{folder_selection}です")

average_options = [f'average{i}' for i in range(1, 11)]
average_selection = st.multiselect(
    '解析する平均値を選択してください:',
    average_options,
    placeholder="平均値を選択してください"
)


if not average_selection:
    st.error('平均値を選択してください。')
else:

    col1, col2, col3 = st.columns(3)
    with col1:
        plot_nyquist = st.checkbox(
            label="ナイキスト線図",
            value=True
        )
        if plot_nyquist and ('tick_nyquist' not in st.session_state):
            st.session_state['tick_nyquist'] = None

    with col2:
        plot_spectre = st.checkbox(
            label="スペクトル",
            value=False
        )

        if plot_spectre and ('tick_spectre' not in st.session_state):
            st.session_state['tick_spectre'] = None

    with col3:
        plot_bode = st.checkbox(
            label="ボード線図",
            value=False
        )

    if not 'key' in st.session_state:
        st.session_state['key'] = 0

    
    x_length = st.sidebar.number_input('実軸の補正値', value=0.0)
    

    real, imag, ch1_spectre, ch2_spectre, frequency, idx_ch1, idx_ch2 = make_graph(average_selection)

    real = [x_length + i for i in real]

    nyquist_container = st.container()
    spectre_container = st.container()
    bode_container = st.container()

    auto_refresh = st.checkbox("自動更新", value=True)

    if auto_refresh or st.button('グラフ作成'):

        if plot_nyquist:
            nyquist_fig = graph_util.plot_graph_1(real, imag, folder_path, st.session_state['key'])
            nyquist_container.pyplot(nyquist_fig)
        else:
            if 'tick_nyquist' in st.session_state:
                del st.session_state['tick_nyquist']
        
        if plot_spectre:
            spectre_fig = graph_util.FFT_graph(ch1_spectre, ch2_spectre, frequency, idx_ch1, idx_ch2, st.session_state['key'])
            spectre_container.pyplot(spectre_fig)
        else:
            if 'tick_spectre' in st.session_state:
                del st.session_state['tick_spectre']

        if plot_bode:
            bode_fig = graph_util.Bode_plot(real, imag, list(frequency.keys()), folder_path)
            bode_container.pyplot(bode_fig)

        st.success('グラフ作成が完了しました。')

        if 'tick_nyquist' in st.session_state:
            state = st.session_state['tick_nyquist']
            x_min = st.sidebar.number_input('X軸の最小値', value=state['x'][0], on_change=update_key)
            x_max = st.sidebar.number_input('X軸の最大値', value=state['x'][1], on_change=update_key)
            y_min = st.sidebar.number_input('Y軸の最小値', value=state['y'][0], on_change=update_key)
            y_max = st.sidebar.number_input('Y軸の最大値', value=state['y'][1], on_change=update_key)

            st.session_state['tick_nyquist'] = {
                'x': (x_min, x_max),
                'y': (y_min, y_max)
            }



        if 'tick_spectre' in st.session_state:
            state = st.session_state['tick_spectre']
            y1_max = st.sidebar.number_input('Ch1のY軸の最大値', value=state['y1'][1], on_change=update_key)
            y2_max = st.sidebar.number_input('Ch2のY軸の最大値', value=state['y2'][1], on_change=update_key)
            # y1_max = st.sidebar.number_input('Ch1のY軸の最大値', value=1.0)
            # y2_max = st.sidebar.number_input('Ch2のY軸の最大値', value=0.2)
            st.session_state['tick_spectre'] = {
                'y1': (0, y1_max),
                'y2': (0, y2_max)
                }
