import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

fs = 48000
hz = 100
n = np.arange(2048)

# Wykres dla zmieszanych sygnałów 100, 300 i 500Hz
trzy_sinusy = (0.5 * np.sin(2 * np.pi * n * hz / fs) +
               0.3 * np.sin(2 * np.pi * n * 3 * hz / fs) +
               0.2 * np.sin(2 * np.pi * n * 5 * hz / fs))

n = np.arange(2048)
trzysin = (np.sin(2 * np.pi * n * 1000 / fs) +
           np.sin(2 * np.pi * n * 3000 / fs) +
           np.sin(2 * np.pi * n * 5000 / fs)) / 3
widmo_trzysin = 20 * np.log10(np.abs(np.fft.rfft(trzysin * np.hamming(2048))) / 1024)
dp_4 = sig.firwin(100, 4000, fs=fs)
trzysin_filtr = sig.lfilter(dp_4, 1, trzysin)
widmo_filtr = 20 * np.log10(np.abs(np.fft.rfft(trzysin_filtr * np.hamming(2048))) / 1024)

f = np.fft.rfftfreq(2048, 1 / fs)
_, ax = plt.subplots(2)
ax[0].plot(f, widmo_trzysin, label='oryginalny')
ax[1].plot(f, widmo_filtr, label='po filtracji', color= "red")
fig = plt.gcf()
fig.set_size_inches(10, 8)
for a in ax:
    a.set_xlabel('Częstotliwość [Hz]')
    a.set_ylabel('Amplituda [dB]')
    a.legend()
    a.set_xlim(0, 10000)
ax[0].set_title('Filtracja sumy trzech sinusów filtrem dolnoprzepustowym')
plt.show()