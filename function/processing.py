import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import librosa

from scipy import signal
from scipy.signal import spectrogram, butter, lfilter, get_window

def AmplitudetoDecibel(mesureInit):
    return 20*np.log10(mesureInit/(1*10**-6))

def DecibeltoAmplitude(mesureInit):
    power = (mesureInit/20 - 6)
    return 10**power

def VectorAmplitudetoDecibel(vector):
    return AmplitudetoDecibel(np.asarray(vector))

def VectorDecibeltoAmplitude(vector):
    return DecibeltoAmplitude(np.asarray(vector))

def MatrixAmplitudetoDecibel(Matrix):
    return AmplitudetoDecibel(np.asarray(Matrix))

def MatrixDecibeltoAmplitude(Matrix):
    return DecibeltoAmplitude(np.asarray(Matrix))

def normalizeRawWaveform(waveform):
    return VectorAmplitudetoDecibel(np.absolute(waveform) + 1*10**-6)*np.sign(waveform)/120

def returnRawWaveform(waveform):
    return (VectorDecibeltoAmplitude(120*np.absolute(waveform)) - 1*10**-6)*np.sign(waveform)

def get_snr(s, noise):
    ns = len(s)
    Ps = np.sum(s*s)/ns
    Pn = np.sum(noise*noise)/ns
    snr = 10*np.log10((Ps - Pn + 0.000001)/(Pn + 0.000001))
    return snr

def coeffBandpassFilter(lowcut, highcut, fs, order=5):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def coeffLowpassFilter(lowcut, fs, order=5):
    nyq = 0.5*fs
    low = lowcut/nyq
    b, a = butter(order, low, btype='low', analog=False)
    return b, a

def bandpassFilter(s, lowcut, highcut, fs, order=5):
    b, a = coeffBandpassFilter(lowcut, highcut, fs, order)
    y = lfilter(b, a, s)
    return y

def lowpassFilter(s, lowcut, fs, order=5):
    b, a = coeffLowpassFilter(lowcut, fs, order)
    y = lfilter(b, a, s)
    return y

def DEMON(s, filter, filterFreq, fs):
    N = len(s)
    if filter=="bande":
        sf = bandpassFilter(s, filterFreq[0], filterFreq[1], fs)
    elif filter=="low":
        sf = lowpassFilter(s, filterFreq[0], fs)
    elif filter=="high":
        sf = lowpassFilter(s, filterFreq[0], fs)
    else:
        sf = s
    w = np.hanning(N)
    yf = 2*np.fft.fft(sf**2*w)/N
    yf[0:int(N/fs/2)] = complex(0,0)
    return yf, N

def DEMONViz(s, filter="bande", filterFreqLow=2000, filterFreqHigh=10000, freq_max_viz=200, fs=22050):
    yf, N = DEMON(s, filter, [filterFreqLow, filterFreqHigh], fs)
    freq_max_viz = fs/2/freq_max_viz
    xf = np.linspace(0.0, fs/2.0/freq_max_viz, int(N/2/freq_max_viz))
    plt.plot(xf, np.abs(yf[0:int(N/2/freq_max_viz)]))
    plt.show()

def MFCC(s, fs=22050):
    nffts = 2048
    nperseg = 512
    n_mels = 256
    n_mfcc = 256
    S = librosa.feature.melspectrogram(y=s, sr=fs, n_fft=2048, hop_length=512, n_mels=n_mels)
    return librosa.feature.mfcc(S=AmplitudetoDecibel(S), n_mfcc=n_mfcc)

def FFTAccumulatedMethod(s, fs, Nbis=64, L=64):
    N = len(s)
    P = int(N/L - Nbis)
    Deltaf = fs/Nbis
    DeltafCycl = fs/(L*P)
    freq = np.linspace(0, Deltaf*Nbis, Nbis)
    freqCycl = np.linspace(Deltaf, DeltafCycl*P + Deltaf, P)
    Xt = np.zeros([Nbis, P])
    h1 = np.hanning(Nbis)
    Xt2 = np.zeros([Nbis, P], dtype=np.complex_)
    for p in range(P):
        Xt[0:Nbis, p] = s[p*L:p*L + Nbis]*h1
    for p in range(P):
        Xt2[0:Nbis, p] = np.fft.fft(Xt[0:Nbis, p])*np.exp(-1j*2*np.pi*freq*(p*L + Nbis/2)/Nbis)
    Yt2 = np.conjugate(Xt2)
    S = np.zeros([Nbis - 1, P], dtype=np.complex_)
    S2 = np.zeros([Nbis - 1, P])
    for n in range(Nbis - 1):
        S[n, :] = Xt2[n, :]*Yt2[n+1, :]
    for n in range(Nbis - 1):
        S2[n, :] = np.real(np.fft.fft(S[n, :]))
    return S2, freq[0:-1], freqCycl

def FAMViz(s, fs, Nbis=64, L=16):
    S, freq, freqCycl = FFTAccumulatedMethod(s, fs, Nbis, L)
    plt.pcolormesh(freqCycl, freq, AmplitudetoDecibel(S + 1*10**-6), cmap='viridis')
    plt.ylabel('frequency (Hz)')
    plt.xlabel('Cyclic frequency (Hz)')
    plt.show()

def cyclicModulationSpectrum(s, fs, nffts = 2048, nperseg = 512):
    N = len(s)
    # nperseg = min(nffts, nperseg*int(N/fs))
    overlap = 0.5
    noverlap = int(nperseg*overlap)
    w = get_window('hanning', nperseg)
    f, t, Sxx = signal.spectrogram(s[0:N], fs=fs, nfft=nffts, nperseg=nperseg, window=w, noverlap=noverlap)
    cms = Sxx
    cms[0:len(f)] = 2*np.fft.fft(abs(Sxx[0:len(f)])**2)/len(t)
    return f, t, cms, N

def cyclicModulationSpectrumViz(s, fs, nffts = 2048, nperseg = 512):
    f, t, cms, N = cyclicModulationSpectrum(s, fs, nffts = 2048, nperseg = 512)
    cyclicFrequencyCut = int(len(t)/2)
    cf = np.linspace(0, int(len(t)/(N/fs)), len(t))
    plt.pcolormesh(cf[:], f, AmplitudetoDecibel(abs(cms[:,:]) + 1*10**-6), cmap='viridis')
    plt.ylabel('frequency (Hz)')
    plt.xlabel('Cyclic frequency (Hz)')
    plt.show()
