import librosa
import librosa
import numpy as np
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# 載入 WAV 檔
file_path = "./wavs/General/Hungry/Hungry_30064.wav"
y, sr = librosa.load(file_path, sr=None, mono=True)
print(sr)

# 取五秒的音訊
duration = 5  # 五秒

y = librosa.util.fix_length(y, int(sr * duration))

# 繪製音訊波形
plt.figure(figsize=(10, 4))
plt.plot(y, color='blue')
plt.title('Audio Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.ylim(-1, 1)  # Set y-axis limits
plt.show()

# 正規化音訊到範圍 [-1, 1]
y_normalized = librosa.util.normalize(y)

# 繪製正規化後的音訊波形
plt.figure(figsize=(10, 4))
plt.plot(y_normalized, color='orange')
plt.title('Normalized Audio Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.ylim(-1, 1)  # Set y-axis limits
plt.show()

# 計算梅爾頻譜 短時傅立葉變換
n_fft = 2048
hop_length = 512

# 定義預強調係數
# alpha = 0.97

# 預強調操作
# y_preemphasized = np.append(y[0], y[1:] - alpha * y[:-1])

# 計算梅爾頻譜
# mel_spec = librosa.feature.melspectrogram(y_preemphasized, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hann', n_mels=128)
mel_spec = librosa.feature.melspectrogram(y_normalized, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hann',n_mels=128)

# 將功率頻譜轉換成分貝刻度
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# 顯示梅爾頻譜圖
librosa.display.specshow(mel_spec_db, y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()
print(mel_spec.shape) #(128, 79)

