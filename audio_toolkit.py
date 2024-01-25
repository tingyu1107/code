import numpy as np
import wave

class AudioLoad(object):
    def __init__(self,filename):
        self._wav_file = wave.open(filename)
        # (nchannels, sampwidth, framerate, nframes, comptype, compname)
        # params = f.getparams()
        self._params = self._wav_file.getparams()
        self._nchannels, self._sample_width, self._frame_rate, self._n_frame = self._params[:4]
        self._signal= np.fromstring(self._wav_file.readframes(self._n_frame), dtype=np.short)

        self._wav_file.close()

    def getMax(self):
        print ("max value = ",self._signal.max())

    def getMin(self):
        print ("min value = ",self._signal.min())

    def getSignal2(self):
        power = self.number_of_powers_of_2()
        return self._signal[:2**power]

    def getSignal(self):
        return self._signal

    def getSignal4s(self,end_idx=40000):
        size = len(self._signal)
        if size < end_idx:
            self._signal = np.append(self._signal,np.zeros(end_idx - size))
        return self._signal[:end_idx]


    def getFrames(self,winlen=8192,overlap=4096):
        res = []

        idx = 0
        while((idx+winlen) < len(self._signal)):
            res.append(np.array(self._signal[idx:idx+winlen],dtype=float))
            idx += overlap

        return res

    def getFrames2(self,winlen=8192,overlap=4096):
        res = []

        idx = 0
        while((idx+winlen) < len(self.getSignal2())):
            res.append(self.getSignal2()[idx:idx+winlen])
            idx += overlap

        return res

    def getSampleRate(self):
        return self._frame_rate

    def wave_save_as(self,signal,filename):
        f = wave.open(filename,mode='wb')
        f.setparams((self._nchannels, self._sample_width, self._frame_rate, self._n_frame, 'NONE', 'noncompressed'))
        f.writeframes(signal)
        f.close()

    def save_as(self,filename):
        f = wave.open(filename,mode='wb')
        f.setparams((self._nchannels, self._sample_width, self._frame_rate, self._n_frame, 'NONE', 'noncompressed'))
        f.writeframes(self._signal)
        f.close()

    def volume_adjustment(self,new_max,is_change_self=False):
        new_signal = (self._signal / self._signal .max()) * new_max

        if is_change_self == True:
            self._signal = new_signal

        return new_signal

    def noise_filter(self,threshold=5000):
        print (type(self._signal))
        for i in range(len(self._signal)):
            if (self._signal[i] < threshold) and (self._signal[i] > (-1*threshold)) :
                self._signal[i] = 0

    def haha(self):
        self._signal = np.fft.fft(self._signal)
        self.noise_filter(threshold=100000)
        self._signal = np.fft.fft(self._signal)

    def number_of_powers_of_2(self):
        size = len(self._signal)
        for i in range(size):
            if 2**i > size:
                break

        return (i-1)

    def normalized_boundary(self,lst):
        x = np.array(lst,dtype=float)
        return x / (max(abs(max(x)),abs(min(x))))

    def preEmphasis(self,a=0.95):
        self._signal = self._pre_emphasis(self._signal,a=a)

    def _pre_emphasis(self,x,a=0.95):
        x2= np.append(x[0],x[1:] - a*x[:-1])
        return x2

