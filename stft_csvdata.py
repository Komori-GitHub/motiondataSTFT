#読み込むファイルを設定ファイルに記載しておくver.
import pandas as pd
from scipy.signal import stft
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
import time

SLIDE=4 #セグメント毎のサンプル数/SLIDE毎に窓関数を掛けてフーリエ変換を行う

# setting.txtからファイル名を読み込む
with open("setting.txt", 'r') as file:
    lines = file.readlines()
    for line in lines:
        if 'sampling' in line or 'freeuency' in line:
            fs = float(line.split(':')[1].strip())#サンプリング周波数読み込み
        if 'num' in line or 'sample' in line or 'segment' in line:
            numperseg = int(line.split(':')[1].strip())#1セグメント毎のサンプル数読み込み
        
# csvファイルからデータを読み込む
df = pd.read_csv("motiondata.csv")

# データを抽出
timedata = df.iloc[:, 0].values
data = df.iloc[:, 1].values

# STFTを行う
frequencies, times, Zxx = stft(data, fs,window='hamming',nperseg=numperseg,noverlap=numperseg-numperseg//SLIDE)



# 各時刻でパワーが最大となる周波数を抜き出す
max_freqs = frequencies[np.argmax(np.abs(Zxx), axis=0)]

# 周期（逆数）を計算
periods = np.where(max_freqs == 0, 0, 1 / max_freqs)

# 結果をデータフレームに保存
results_df = pd.DataFrame({
    'Time': timedata[0:len(timedata):numperseg//SLIDE],
    'Frequency': max_freqs[0:len(timedata[0:len(timedata):numperseg//SLIDE])],
    'Period': periods[0:len(timedata[0:len(timedata):numperseg//SLIDE])]
})


# CSVファイルに書き込む
results_df.to_csv('output.csv', index=False)
# テキストファイルに書き込む
txtf= open("period.txt","w")
txtf.write("mode[s]:{}\n".format(mode(periods).mode))
txtf.write("median[s]:{}\n".format(np.median(periods)))
txtf.write("average[s]:{}\n".format(np.mean(periods)))
txtf.close()

# スペクトルグラムをプロット
plt.figure(figsize=(8, 4))
spec = plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
plt.colorbar(spec, label='Magnitude')# カラーバーを追加
plt.title('STFT Spectrogram')# タイトルと軸ラベルを設定
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0,1)
plt.show(block=False)

# 動揺周期をプロットする
plt.figure(figsize=(8, 4))
plt.plot(timedata[0:len(timedata):numperseg//SLIDE],periods[0:len(timedata[0:len(timedata):numperseg//SLIDE])])
plt.title('STFT result Period')
plt.ylabel('Period [s]')
plt.xlabel('Time [sec]')
plt.ylim(0,50)
plt.show(block=False)
plt.show() # 描画を更新

print("最頻値[s]:",mode(periods).mode)
print("中央値[s]:",np.median(periods))
print("平均値[s]:",np.mean(periods))