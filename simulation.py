import os

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

from get_data import samples_dir


def callback_mix(premix, snr=0, ref_mic=0, n_src=None):
    """
    所望のSNRで背景雑音を重畳し、音源を混合する関数

    Parameters
    ----------
    premix: ndarray (n_src, n_mic, n_samples)
        混合前の音源信号
    snr: float
        背景雑音とのSNR
    ref_mic: int
        `ref_mic` 番目のマイクにおける信号を正規化する（SNR=0にする）
    n_src: int
        音源数

    Returns
    -------
    mix: ndarray (n_mic, n_samples)
        `premix` をSN比 `snr` で混合した信号
    """
    # first normalize all separate recording
    # to have unit power at microphone one
    p_mic_ref = np.std(premix[:, ref_mic, :], axis=1)
    premix /= p_mic_ref[:, None, None]

    # compute noise variance
    sigma_n = np.sqrt(10 ** (-snr / 10))

    # Mix down the recorded signals
    mix = np.sum(premix[:n_src, :], axis=0) + sigma_n * np.random.randn(
        *premix.shape[1:]
    )

    return mix


# 使用する音声ファイルの場所
n_src = 3
wav_files = [
    f"{samples_dir}/cmu_arctic_female_clb_7.wav",
    f"{samples_dir}/cmu_arctic_male_ahw_7.wav",
    f"{samples_dir}/cmu_arctic_female_eey_9.wav",
]

# 音声ファイルの読み込み
wavs = [wavfile.read(f)[1] / 2 ** 15 for f in wav_files]

# pyroomacoustics でシミュレーション

# 部屋の大きさ [m]: (x, y, z)
room_dim = np.array([6.0, 4.0, 3.5])

# パラメータ
room_params = {
    "fs": 16000,  # サンプリング周波数
    "absorption": 0.35,  # 壁面の吸音率
    "max_order": 17,  # 鏡像法の計算次数
}

# 部屋の定義
room = pra.ShoeBox(p=room_dim, **room_params)

# 鏡像法では部屋の中心にマイクや音源を配置するとノイズが発生する
# これを避けるためにランダムな値を足してずらす
np.random.seed(0)
center = room_dim / 2 + np.random.rand(*room_dim.shape) * 0.01

# マイクの位置
# 半径10cmの円状マイクロホンアレイ
mic_locs_xy = pra.beamforming.circular_2D_array(
    center[:2], M=n_src, phi0=np.random.rand() * 0.01, radius=0.10
)
mic_locs_z = np.array([room_dim[2] / 2 for i in range(n_src)])
mic_locs = np.vstack([mic_locs_xy, mic_locs_z])

# マイクを配置
room.add_microphone_array(pra.MicrophoneArray(mic_locs, room.fs))

# 音源の位置
# ここでは部屋の中心から2mの円周上に均等に配置している
# これをいじって色々試してみるといいです
r = np.minimum(*room_dim[:2]) * 0.4
deg = [k * 360 / n_src for k in range(n_src)]
rad = [d / 180 * np.pi for d in deg]
src_x = [r * np.cos(t) for t in rad]
src_y = [r * np.sin(t) for t in rad]
src_locs = [
    [center[0] + src_x[i], center[1] + src_y[i], center[2]] for i in range(n_src)
]

# 音源の配置
for sl, a in zip(src_locs, wavs):
    room.add_source(sl, signal=a)

# 音源、マイクの配置を描画
out_dir = f"./output"
os.makedirs(out_dir, exist_ok=True)
fig, ax = room.plot(img_order=0)
fig.savefig(f"{out_dir}/room_layout.png")

# 室内インパルス応答の畳み込み
room.compute_rir()

# 残響時間の測定
measured_rt60 = pra.experimental.measure_rt60(room.rir[0][0], room.fs)
print("RT60 is approximately", measured_rt60)

# Run the simulation
# shape: (source, microphone, data)
callback_mix_kwargs = {
    "ref_mic": 0,
    # 背景雑音のSNR（前に渡したサンプルではここが30dBにしていたので少し背景雑音が大きかったです。）
    "snr": 90,
}
premix = room.simulate(
    callback_mix=callback_mix,
    callback_mix_kwargs=callback_mix_kwargs,
    return_premix=True,
)
premix /= np.max(np.abs(premix))
reference = premix[:, callback_mix_kwargs["ref_mic"], :]

# 混合音、参照信号を保存
for i in range(n_src):
    wavfile.write(f"{out_dir}/ref{i}.wav", room.fs, reference[i])
    wavfile.write(f"{out_dir}/mix{i}.wav", room.fs, room.mic_array.signals[i])
