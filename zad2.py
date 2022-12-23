import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.io import wavfile

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

samplerate, data = wavfile.read('file_example_WAV_1MG.wav')

# Wykres dla sygnału 100Hz
fs = samplerate
hz = 100
n = np.arange(2048)

# Wykres dla zmieszanych sygnałów 100, 300 i 500Hz


widmo_trzech_sinusow = np.fft.rfft(data)
f = np.fft.rfftfreq(len(data)*2-1, 1 / fs)
plt.plot(f, np.abs(widmo_trzech_sinusow) / 1024)
plt.xlim(0, 1000)
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda widma')
plt.title('Widmo trzech sinusów- przed filtrem dolnoprzepustowym')
plt.show()

lowcut = 200.0
highcut = 550.0
cutoff_frequency = 2000
lp = butter_bandpass_filter(data, lowcut, highcut, fs)
lp = np.fft.rfft(lp)
plt.clf()
plt.plot(f, np.abs(lp) / 1024)
plt.xlim(0, 1000)
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda widma')
plt.title('Widmo trzech sinusów- po filtrze dolnoprzepustowym')
plt.show()