import os

import numpy as np
import pyroomacoustics as pra
from mir_eval.separation import bss_eval_sources
from scipy.io import wavfile

# 使用する混合音の場所
# 各ファイルは1ch、長さの等しい信号
mix_files = [
    "./output/mix0.wav",
    "./output/mix1.wav",
    "./output/mix2.wav",
]
n_src = len(mix_files)

# 混合音の読み込み
# (n_src, n_samples)
fs = wavfile.read(mix_files[0])[0]
mix = np.array([wavfile.read(f)[1].astype(np.float32) for f in mix_files])

# 参照信号の読み込み
# (n_src, n_samples)
mix_files = [
    "./output/ref0.wav",
    "./output/ref1.wav",
    "./output/ref2.wav",
]
ref = np.array([wavfile.read(f)[1].astype(np.float32) for f in mix_files])

# STFT parameters
n_fft = 4096
hop = n_fft // 2
win_a = pra.hamming(n_fft)
win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

# STFT
X = pra.transform.stft.analysis(mix.T, n_fft, hop, win=win_a)

# 音源分離
Y = pra.bss.auxiva(X, n_iter=50, proj_back=True)

# 逆STFT
y = pra.transform.stft.synthesis(Y, n_fft, hop, win=win_s)
y = y[n_fft - hop :, :].T

# pyroomacoustics でシミュレーションした信号は、
# 室内インパルス応答を畳み込むため、混合前の信号と長さが変わる
# 分離性能を評価するためには長さが揃っていないといけないため、小さい方に合わせる
m = np.minimum(y.shape[1], ref.shape[1])

# 分離性能評価
SDR0, SIR0, _, _ = bss_eval_sources(ref[:, :m], mix[:, :m])
SDR, SIR, _, perm = bss_eval_sources(ref[:, :m], y[:, :m])
print("Before BSS")
print("SDR:", SDR0)
print("SIR:", SIR0)
print("After BSS")
print("SDR:", SDR)
print("SIR:", SIR)

# 分離音のパーミュテーションを揃える
y = y[perm]

# 分離音を保存
out_dir = f"./output"
os.makedirs(out_dir, exist_ok=True)
for i in range(n_src):
    wavfile.write(f"{out_dir}/sep{i}.wav", fs, y[i])
