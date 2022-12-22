import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def lowpass(signal):
    dp_2 = sig.firwin(100, 10000, fs=fs)
    chirp_dp = sig.lfilter(dp_2, 1, signal)
    return chirp_dp

# Wykres dla sygnału 100Hz
fs = 48000
hz = 100
n = np.arange(2048)

# Wykres dla zmieszanych sygnałów 100, 300 i 500Hz
three_sinuses = (0.5 * np.sin(2 * np.pi * n * hz / fs) +
                 0.3 * np.sin(2 * np.pi * n * 3 * hz / fs) +
                 0.2 * np.sin(2 * np.pi * n * 5 * hz / fs))

widmo_trzech_sinusow = np.fft.rfft(three_sinuses)
f = np.fft.rfftfreq(2048, 1 / fs)
plt.plot(f, np.abs(widmo_trzech_sinusow) / 1024)
plt.xlim(0, 1000)
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda widma')
plt.title('Widmo trzech sinusów- przed filtrem dolnoprzepustowym')
plt.show()

cutoff_frequency = 2000
lp = lowpass(three_sinuses)
lp = np.fft.rfft(lp)
plt.plot(f, np.abs(lp) / 1024)
plt.xlim(0, 1000)
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda widma')
plt.title('Widmo trzech sinusów- po filtrze dolnoprzepustowym')
plt.show()
