import math
import numpy as np
from scipy.signal import chirp, hilbert
from scipy import linalg, fft as sp_fft
import matplotlib
import matplotlib.pyplot as plt
import wave

wavf = './test.wav'
wr = wave.open(wavf, 'r')
data = wr.readframes(wr.getnframes())
wr.close()
sig = np.frombuffer(data, 'int16')
fs = wr.getframerate()
t = np.arange(len(sig)) / fs

# fs = 400
# dt = 1./fs
# samples = int(fs * 1)
# t = np.arange(samples) / fs
# sig = chirp(t, 20.0, t[-1], 40.0)
# sig *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )


def myhilbert(sig):
    x = np.asarray(sig)
    N = x.shape[-1]
    Xf = np.fft.fft(sig, N, axis=-1)
    # print('fft')
    # print(Xf)
    h = np.zeros(N)
    # print('ifft')
    # print(sp_fft.ifft(Xf, axis=-1))
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N // 2) // 2] = 2
    return sp_fft.ifft(Xf * h, axis=-1)

def my_angle(sig):
    return [math.atan2(data.imag, data.real) for data in sig]

def diff(sig):
    result = []
    x1 = sig[0]
    for x2 in sig[1:]:
        result.append(fs * (x2 - x1) / (2 * np.pi))
        x1 = x2
    return result

def unwrap(sig):
    for i in range(1, len(sig)):
        div = sig[i] - sig[i - 1]
        if not (div < -np.pi or np.pi < div):
            continue
        for j in range(i, len(sig)):
            if div < -np.pi:
                sig[j] += 2 * np.pi
            elif div > np.pi:
                sig[j] -= 2 * np.pi
    return sig
print(sig)
#analytic_signal = myhilbert(sig)
#print(analytic_signal)
#print('angle')
#print(np.angle(analytic_signal))
#print('unwrap')
#print(np.unwrap(np.angle(analytic_signal)))
#print('diff')
#plt.plot(np.diff(np.unwrap(np.angle(analytic_signal))) / (2 * np.pi) * fs)

# 自前
analytic_signal = myhilbert(sig)
angles = my_angle(analytic_signal)
print(angles)
unwrap_data = unwrap(angles)
# print(unwrap_data)
# print('unwrap')
diff_data = diff(angles)
print(diff_data)
plt.plot(diff_data)
# numpyよくつかったやつ

# plt.plot(t[63000:len(sig)//2], (np.diff(np.unwrap(np.angle(analytic_signal))) / (2 * np.pi) * fs)[63000:len(sig)//2])

plt.savefig('freq.jpg')
