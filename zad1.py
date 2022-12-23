import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def lowpass_filter(signal, sample_rate, cutoff_frequency):
  # Obliczenie liczby próbek w sygnale
  num_samples = signal.shape[0]
  # Obliczenie odstępu pomiędzy kolejnymi próbkami sygnału
  sample_spacing = 1.0 / sample_rate
  # Obliczenie częstotliwości krokowej
  frequency_step = 1.0 / (num_samples * sample_spacing)
  # Obliczenie częstotliwości skali
  frequencies = np.arange(num_samples) * frequency_step
  # Obliczenie FFT
  fft = np.fft.fft(signal)
  # Utworzenie maski filtrującej
  mask = frequencies < cutoff_frequency
  # Zastosowanie maski do FFT
  filtered_fft = fft.copy()
  filtered_fft[mask] = 0
  # Odtworzenie sygnału po zastosowaniu filtru
  filtered_signal = np.fft.ifft(filtered_fft)
  return filtered_signal


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

lp = lowpass_filter(three_sinuses, fs, 47600)
lp = np.fft.rfft(lp)
plt.plot(f, np.abs(lp) / 1024)
plt.xlim(0, 1000)
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda widma')
plt.title('Widmo trzech sinusów- po filtrze dolnoprzepustowym')
plt.show()
