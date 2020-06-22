import numpy as np
import pylab
import time

import matplotlib.pyplot as plt
import librosa
import librosa.display

from function.processing import *

def signal_visualization(s, title, fs=22050):
    scale = max(abs(s))
    N = len(s)
    dt = N/fs
    s = s.reshape((N))
    print("signal of " + str(dt) + " sec sampled at " + str(fs) + " Hz.")
    fileName = "display/" + title.replace(" ", "_")

    ################### Temporal ###################

    tf = np.linspace(0.0, dt, N)
    pylab.plot(tf, s[0:N])
    pylab.title(str(title))
    pylab.grid()
    pylab.ylabel('signal magnitude (Pa)')
    pylab.xlabel('time (sec)')
    pylab.savefig(str(fileName) + "_Temporal" + ".png", dpi=200)
    pylab.close('all')

    ################### FFT ###################

    yf = 2*np.fft.fft(s)/N
    xf = np.linspace(0.0, fs/2.0, int(N/2))
    pylab.plot(xf, np.abs(yf[0:int(N/2)]))
    pylab.title(str(title) + " FFT")
    pylab.grid()
    pylab.ylabel('FFT magnitude (Pa)')
    pylab.xlabel('frequency (Hz)')
    pylab.savefig(str(fileName) + "_FFT" + ".png", dpi=200)
    pylab.close('all')

    ################### Power spectral density ###################

    yf = AmplitudetoDecibel(yf + 1*10**-6)
    pylab.plot(xf, np.real(yf[0:int(N/2)]))
    pylab.title(str(title) + " Power Spectral Density")
    pylab.grid()
    pylab.ylabel('FFT magnitude (dB)')
    pylab.xlabel('frequency (Hz)')
    pylab.savefig(str(fileName) + "_Power_Spectral_Density" + ".png", dpi=200)
    pylab.close('all')

    ################### Spectrogram ###################

    nffts = 2048
    nperseg = 512
    overlap = 0.5
    noverlap = int(nperseg*overlap)
    w = get_window('hanning', nperseg)
    f, t, Sxx = signal.spectrogram(s[0:N], fs=fs, nfft=nffts, nperseg=nperseg, window=w, noverlap=noverlap)
    plt.pcolormesh(t, f, Sxx, cmap='viridis')
    plt.title(str(title) + " Spectrogram")
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time (sec)')
    plt.colorbar()
    plt.savefig(str(fileName) + "_Spectrogram" + ".png", dpi=200)
    plt.close('all')

    ################### Spectrogram logarithmic scale ###################

    Sxx2 = AmplitudetoDecibel(Sxx + 1*10**-6)
    plt.pcolormesh(t, f, Sxx2, cmap='bone')
    plt.title(str(title) + " Spectrogram dB")
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time (sec)')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(str(fileName) + "_Spectrogram_dB" + ".png", dpi=200)
    plt.close('all')

    ################### MFCC ###################

    Sxx3 = MFCC(s, fs)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(Sxx3, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.savefig(str(fileName) + "_MFCC" + ".png", dpi=200)
    plt.close('all')

    ################### DEMON ###################

    filter="low"
    filterFreqLow=1000
    filterFreqHigh=10000
    freq_max_viz=200
    yf, N = DEMON(s, filter, [filterFreqLow, filterFreqHigh], fs)
    freq_max_viz = fs/2/freq_max_viz
    xf = np.linspace(0.0, fs/2.0/freq_max_viz, int(N/2/freq_max_viz))
    plt.plot(xf, np.abs(yf[0:int(N/2/freq_max_viz)]))
    plt.title('DEMON')
    plt.savefig(str(fileName) + "_DEMON" + ".png", dpi=200)
