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


shunt_registor= st.selectbox(
    'シャント抵抗を選択してください(Ω)',
    [1, 0.1, 0.01],
    index = None,
    placeholder="選択してください")

vector= st.selectbox(
    '電流の向きを選んでください',
    [1.0, -1.0],
    index = None,
    placeholder="選択してください")


if st.button('キャッシュをクリア'):
    st.cache_data.clear()
    st.success('キャッシュをクリアしました。')


# 書き込み可能なベースディレクトリを設定
base_dir = '/Users/ken/Library/CloudStorage/Box-Box/片山研究室・共有フォルダ/1_研究データ/1_2024_LIB_EIS_ML/4_パルス幅を変化させてEIS'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# フォルダ名の入力
folder_name = st.text_input('作成するフォルダの名前を入力してください:')

if st.button('フォルダを作成'):
    # 入力されたフォルダ名を基にフォルダを作成
    if folder_name:
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            st.success(f"フォルダ '{folder_name}' が作成されました。")
        else:
            st.warning(f"フォルダ '{folder_name}' は既に存在します。")
    else:
        st.error('フォルダ名を入力してください。')

# # 既存のフォルダ内のファイルを表示
# existing_folder = st.text_input('表示したいフォルダの名前を入力してください:')

# if st.button('フォルダ内のファイルを表示'):
#     if existing_folder:
#         folder_path = os.path.join(base_dir, existing_folder)
#         if os.path.exists(folder_path):
#             files = os.listdir(folder_path)
#             if files:
#                 st.write(f"フォルダ '{existing_folder}' 内のファイル:")
#                 for file in files:
#                     st.write(file)
#             else:
#                 st.write(f"フォルダ '{existing_folder}' 内にファイルがありません。")
#         else:
#             st.error(f"フォルダ '{existing_folder}' は存在しません。")
#     else:
#         st.error('フォルダ名を入力してください。')

# サブフォルダの作成
if folder_name:
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(os.path.join(folder_path, 'result'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'インピーダンスアナライザ'), exist_ok=True)

st.set_option('deprecation.showPyplotGlobalUse', False)



# セッションステートを初期化
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("オシロスコープのデータをアップロード", accept_multiple_files=True)
with col2:
    analyzer_uploaded_files = st.file_uploader("インピーダンスアナライザのデータをアップロード", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # アップロードされたファイルをセッションステートに保存
        st.session_state['uploaded_files'].append(uploaded_file)
        # ファイルを保存
        file_path = os.path.join(folder_path, 'result', uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

if analyzer_uploaded_files:
    for analyzer_uploaded_file in analyzer_uploaded_files:
        # アップロードされたファイルをセッションステートに保存
        st.session_state['uploaded_files'].append(analyzer_uploaded_file)
        # ファイルを保存
        file_path = os.path.join(folder_path, 'インピーダンスアナライザ', analyzer_uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(analyzer_uploaded_file.getbuffer())

# フォルダの選択
folder_selection = st.selectbox(
    '解析するフォルダを選択してください:',
    [f.name for f in pathlib.Path(base_dir).iterdir() if f.is_dir()],
    index=0,
    placeholder="フォルダを選択してください"
)
folder_path = os.path.join(base_dir, folder_selection)
st.write(f"選択したフォルダは{folder_selection}です")

@st.cache_data
def make_graph():
    data_path = pathlib.Path(os.path.join(folder_path, 'result'))
    freq_list = list(
        set(
            map(
                lambda p: int(str(p.stem).split("-")[0].replace("frequency", "")),
                data_path.glob("*.CSV"),
            )
        )
    )
    print(freq_list)
    real, imag = [], []
    idx_ch1, idx_ch2 = {}, {}
    frequency, ch1_spectre, ch2_spectre = {}, {}, {}

    for f in natsorted(freq_list):
        impedance_list = []
        for p in data_path.glob(f"frequency{f}-*"):
            ch1_buff, ch2_buff = [], []
            # Sampling period
            Ts = float(pd.read_csv(p, nrows=20).loc["Sampling Period"].iat[0, 1])
            # Load data (ch1: current, ch2: voltage)
            meas_data = pd.read_csv(p, skiprows=25)
            # Apply window function
            window_f = windows.blackman(len(meas_data))
            meas_data.iloc[:, 1] = meas_data.iloc[:, 1] * window_f
            meas_data.iloc[:, 3] = meas_data.iloc[:, 3] * (1/shunt_registor) * window_f * vector
            # Fourier transform
            freq = np.fft.rfftfreq(len(meas_data), d=Ts)[1:]
            F_ch1 = np.fft.rfft(meas_data.iloc[:, 1].to_numpy())[1:]
            F_ch2 = np.fft.rfft(meas_data.iloc[:, 3].to_numpy())[1:]
            # Extract near target frequency
            freq_lower = f * 0.85
            freq_upper = f * 1.15
            # Get max spectre
            F_ch1_ = F_ch1.copy()
            F_ch2_ = F_ch2.copy()
            F_ch1_[~((freq_lower < freq) & (freq < freq_upper))] = 0
            F_ch2_[~((freq_lower < freq) & (freq < freq_upper))] = 0
            max_spectre_idx_v = np.argmax(np.abs(F_ch1_))
            max_spectre_idx_i = np.argmax(np.abs(F_ch2_))
            idx_ch1[f] = max_spectre_idx_v
            idx_ch2[f] = max_spectre_idx_i
            N = len(meas_data)
            # Calculate impedance
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
    
    # Set axis ranges based on sliders
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel('Real Part of Impedance')
    ax.set_ylabel('Imaginary Part of Impedance')
    ax.set_title('Impedance Analysis')
    ax.legend()
    # st.pyplot(fig)
    plot_container.pyplot(fig) 

def plot_graph(frequency, real, imag):
    fig = go.Figure()

    # Plot measured data
    fig.add_trace(go.Scatter(
        x=real,
        y=imag,
        mode='markers',
        name='Measured Data',
        marker=dict(size=6),
        hovertemplate='Real: %{x:.3f}<br>Imaginary: %{y:.3f}<br>Frequency: %{text}<extra></extra>',
        text=[f'{f} Hz' for f in frequency]  # Frequency情報を表示
    ))
    analyzer_data_path = pathlib.Path(os.path.join(folder_path, 'インピーダンスアナライザ'))
    analyzer_freq_list = list(
        set(
                analyzer_data_path.glob("*.csv")
        )
    )

    # Plot analyzer data
    # analyzer_data_path = pathlib.Path("./インピーダンスアナライザ")
    # analyzer_freq_list = list(analyzer_data_path.glob("*.csv"))
    for f in analyzer_freq_list:
        analyzer_df = pd.read_csv(f, skiprows=18)
        analyzer_ch1 = analyzer_df.iloc[0:21, 2]
        analyzer_ch2 = -1 * analyzer_df.iloc[0:21, 3]
        fig.add_trace(go.Scatter(
            x=analyzer_ch1,
            y=analyzer_ch2,
            mode='markers',
            name='Analyzer Data'
        ))

    fig.update_layout(
        title='Impedance Analysis',
        xaxis_title='Real Part of Impedance',
        yaxis_title='Imaginary Part of Impedance',
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        hovermode='closest'
    )

    st.plotly_chart(fig)


def FFT_graph(ch1_spectre, ch2_spectre, frequency, idx_ch1, idx_ch2, y1_max, y2_max, plot_container):
    fig = plt.figure(figsize=(20, 10))


    ax_ch1 = fig.add_subplot(121)
    ax_ch1.set_xlim((1, 10**4))
    ax_ch1.set_ylim(0, y1_max)
    ax_ch1.set_xscale("log")
    ax_ch1.set_ylabel('current[A]')
    ax_ch1.set_xlabel('frequency[Hz]')

    for k, v in ch1_spectre.items():
        ax_ch1.plot([k, k], [0, 0.05], c=cm.jet(
            (k - 10) / (1000 - 10)), linestyle="dashed")
        ax_ch1.plot(frequency[k], v, label=f"{k} Hz", c=cm.jet((k - 10) / (1000 - 10)))
        ax_ch1.scatter(
            [frequency[k][idx_ch1[k]]],
            v[idx_ch1[k]],
            marker="*",
            color=cm.jet((k - 10) / (1000 - 10)),
        )

    ax_ch2 = fig.add_subplot(122)
    ax_ch2.set_xlim((1, 10**4))
    ax_ch2.set_ylim(0, y2_max)
    ax_ch2.set_xscale("log")
    ax_ch2.set_ylabel('voltage[V]')
    ax_ch2.set_xlabel('frequency[Hz]')


    for k, v in ch2_spectre.items():
        ax_ch2.plot([k, k], [0, 0.3], c=cm.jet(
            (k - 10) / (1000 - 10)), linestyle="dashed")
        ax_ch2.plot(frequency[k], v, label=f"{k} Hz", c=cm.jet((k - 10) / (1000 - 10)))
        ax_ch2.scatter(
            [frequency[k][idx_ch2[k]]],
            v[idx_ch2[k]],
            marker="*",
            color=cm.jet((k - 10) / (1000 - 10)),
        )

    ax_ch1.legend()
    ax_ch2.legend()   
    # st.pyplot(fig) 
    plot_container.pyplot(fig) 

st.sidebar.title("軸の調整")
x_min = st.sidebar.slider('x軸の最小値', min_value=-0.030, max_value=0.1, value=0.030, step=0.005)
x_max = st.sidebar.slider('x軸の最大値', min_value=0.050, max_value=0.1, value=0.10, step=0.005)
y_min = st.sidebar.slider('y軸の最小値', min_value=-0.01, max_value=0.0, value=0.0, step=0.005)
y_max = st.sidebar.slider('y軸の最大値', min_value=0.0, max_value=0.050, value=0.010, step=0.005)
y1_max = st.sidebar.slider('電圧の最大値', min_value=0.00, max_value=0.015, value=0.010, step=0.005)
y2_max = st.sidebar.slider('電流の最大値', min_value=0.00, max_value=0.3, value=0.20, step=0.01)

# if st.session_state['uploaded_files']:
st.write("保持されたデータを使用:")
real, imag, ch1_spectre, ch2_spectre, frequency, idx_ch1, idx_ch2 = make_graph()
plot_container = st.empty()  # プレースホルダを作成

if st.button('インピーダンス軌跡を生成'):
    plot_graph(frequency, real, imag)
    plot_graph_1(real, imag, x_min, x_max, y_min, y_max, plot_container)

if st.button('周波数解析を行う'):
    FFT_graph(ch1_spectre, ch2_spectre, frequency, idx_ch1, idx_ch2, y1_max, y2_max, plot_container)


