import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy.signal as sig


wav_fs, klarnet = wavfile.read('klarnet.wav')
f, t, Sxx = sig.spectrogram(klarnet, fs=wav_fs, window=np.hamming(2048), nperseg=2048, noverlap=1536,
                            scaling='spectrum', mode='magnitude')

widmo_klarnet = 20 * np.log10(np.abs(np.fft.rfft(klarnet * np.hamming(69124))) / 1024)


minimum = 3000
maksimum = 4000
f = np.fft.rfftfreq(69124, 1 / wav_fs)
pp_24 = sig.firwin(100, (minimum, maksimum), pass_zero=False, fs=wav_fs)
klarnet_filtrowany = sig.lfilter(pp_24, 1, klarnet)
widmo_filtr_pp = 20 * np.log10(np.abs(np.fft.rfft(klarnet_filtrowany * np.hamming(69124))) / 1024)


_, ax = plt.subplots(2)
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax[0].plot(f, widmo_klarnet, label='oryginalny')
ax[1].plot(f, widmo_filtr_pp, label='po filtracji')
for a in ax:
    a.set_xlabel('częstotliwość [Hz]')
    a.set_ylabel('amplituda [dB]')
    a.legend()
    a.set_xlim(0, 8000)
    a.set_ylim(0, 150)
ax[0].set_title(f"Filtr pasmowo-przepustowy od {minimum}Hz do {maksimum}Hz")
plt.show()

x= []
for i in range(len(klarnet)):
    x.append(i/wav_fs)

_, ax = plt.subplots(2)
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax[0].plot(x, klarnet/1000, label='oryginalny')
ax[1].plot(x, klarnet_filtrowany/1000, label='po filtracji')

for a in ax:
    a.set_xlabel('Czas [s]')
    a.set_ylabel('amplituda [dB]')
    a.legend()
    a.set_xlim(0, 1.6)
    a.set_ylim(-30, 30)
ax[0].set_title('Filtr pasmowo-przepustowy od {minimum}Hz do {maksimum}Hz')
plt.show()