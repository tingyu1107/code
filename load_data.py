import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
import librosa
import numpy as np
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt
#process or data process
import numpy as np
from numpy.fft import fft
from scipy.signal import resample
from scipy import signal as scipy_signal
import re
import random
import cv2
from audio_toolkit import *



def wav2features_mfcc_KTeer(file_path):

    y,sr = librosa.load(file_path,sr=8000)

    if(len(y)<40000):
        y = np.pad(y,(0,40000-len(y)),'mean')
    else:
        y = y[:40000]

    y = librosa.util.normalize(y)


    hop_length = 64
    frame_length = 128

    ### ST energy ###
    energy = np.array([
        sum(abs(y[i:i+frame_length]**2)) for i in range(0, len(y)+1, hop_length)
    ],dtype='float32')


    ### RMSE ###
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length, win_length=frame_length, window='hann', center=True, pad_mode='reflect')
    S = librosa.magphase(stft)[0]
    rmse = librosa.feature.rms(S=S, frame_length=frame_length, hop_length=hop_length, center=True)
        
    ### ZCR ###
    zcr = np.array([
        sum(abs(np.diff(np.sign(y[i:i+frame_length]))))/2
        for i in range(0, len(y)+1, hop_length)
    ],dtype='float32')


    rmse = rmse.flatten()
    zcr = zcr.flatten()

    frameIndex = np.asarray(np.where((energy > 0.35) & (rmse > 0.005) & (zcr > 30)))*hop_length

    filterY = np.zeros(len(y))
    for j in frameIndex[0]:
        for i in range(len(y)):
            if i in range(j,j+frame_length):
                filterY[i] = y[i]

    mfcc = librosa.feature.mfcc(y=filterY, sr=sr, S=S, n_mfcc=48)

    mfcc = mfcc.reshape(626,48,1)

    return mfcc

def wav2feature_jiajing(file_path):

    img_rows, img_cols = 227, 227
    resize_photo = (img_rows, img_cols)


    image = cv2.imread(file_path)
    img = cv2.resize(image, (resize_photo))

    return img

def wav2features_lungyo(file_path):
    audio = AudioLoad(file_path)
    audio.preEmphasis()
    s = audio.getSignal4s()
    signal = audio.normalized_boundary(s)
    res = []
    
    resize_int=8000
    res.append(resample(signal,(resize_int)))
    feature = np.array(res,dtype=float)
    return feature

def wav2features_1D(file_path):
    y, sr = librosa.load(file_path, mono=True, sr=None)
    y = librosa.util.normalize(y)

    if(len(y)<40000):
        y = np.pad(y,(0,40000-len(y)),'mean')
    else:
        y = y[:40000]
    return y

def wav2features_mfcc_IICC(file_path):
    y, sr = librosa.load(file_path, mono=True, sr=None)
    y = librosa.util.normalize(y)

    if(len(y)<40000):
        y = np.pad(y,(0,40000-len(y)),'mean')
    else:
        y = y[:40000]
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    return mfcc

def wav2features_mfcc(file_path):
    max_len=79
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = librosa.util.normalize(wave)
    mfcc = librosa.feature.mfcc(wave, sr=8000, n_mfcc=13, fmin=250)

    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        feature = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        feature = mfcc[:, :max_len]

    delta1 = librosa.feature.delta(feature,1)
    delta2 = librosa.feature.delta(feature,2)
    feature = np.concatenate((feature,delta1),axis=0)
    feature = np.concatenate((feature,delta2),axis=0)
    feature = feature.reshape(1,39,79,1)

    return feature

def wav2features_melspectrogram(file_path):
    y, sr = librosa.load(file_path, mono=True, sr=None)
    # y = librosa.util.normalize(y)

    if(len(y)<40000):
        y = np.pad(y,(0,40000-len(y)),'mean')
    else:
        y = y[:40000]

    melspectrogram = librosa.feature.melspectrogram(y, sr=8000)
    logmelspec = librosa.power_to_db(melspectrogram)

    # 顯示梅爾頻譜圖
    # librosa.display.specshow(logmelspec, y_axis='mel', fmax=8000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel Spectrogram')
    # plt.show()
    return melspectrogram

def wav2features_zcr(file_path):
    y, sr = librosa.load(file_path, mono=True, sr=None)
    y = librosa.util.normalize(y)

    if(len(y)<40000):
        y = np.pad(y,(0,40000-len(y)),'mean')
    else:
        y = y[:40000]

    zcr = librosa.feature.zero_crossing_rate(y=y)

    return zcr

def get_labels(path):
    labels = os.listdir(path)
    labels = sorted(labels)
    label_indices = np.arange(0, len(labels))

    return labels, label_indices, to_categorical(label_indices)

def load_data_to_array(path):
    labels, _, label_oneHot = get_labels(path)
    waves_vectors = []
    label_array = []
    wav_num = []
    wav_num.append(0)

    file_name = []
    file_sort = []

    for index, label in enumerate(labels):
        wavfiles = [ os.path.join(path,label,wavfile) for wavfile in os.listdir(path + '/' + label)]
        
        wavfiles = sorted(wavfiles)
        wav_num.append(len(wavfiles))
        for wavfile in tqdm(wavfiles, "Processing of label - '{}'".format(label)):
            file_name.append(wavfile)

            # feature = wav2features_mfcc_IICC(wavfile)#IICC
            # feature = wav2features_lungyo(wavfile)
            # feature = wav2feature_jiajing(wavfile)
            # feature = wav2features_mfcc_KTeer(wavfile)

            ### load KTeer npy ###
            feature = np.load(wavfile)
            waves_vectors.extend(feature)

            for i in range(len(feature)):
                label_array.append(index)
            ### load KTeer npy ###

            # waves_vectors.append(feature)
            # label_array.append(index)

    X = np.array(waves_vectors) 
    Y = to_categorical(label_array)

    return X,Y,wav_num

def get_train_test(path):

    split_ratio=0.8
    random_state=42
    X,Y,_ = load_data_to_array(path)

    return train_test_split(X, Y, test_size= (1 - split_ratio), shuffle=True, stratify=Y, random_state=random_state)

def get_testWav(path):
    
    X,Y,wav_num = load_data_to_array(path)

    return X, Y, wav_num

def load_data_to_array_personal(path):
    labels, _, label_oneHot = get_labels(path)
    waves_vectors = []
    waves_vectors_mel = []

    label_array = []
    wav_num = []
    wav_num.append(0)

    file_name = []
    file_sort = []

    for index, label in enumerate(labels):
        wavfiles = [ os.path.join(path,label,wavfile) for wavfile in os.listdir(path + '/' + label)]
        
        wavfiles = sorted(wavfiles)
        wav_num.append(len(wavfiles))
        for wavfile in tqdm(wavfiles, "Processing of label - '{}'".format(label)):
            file_name.append(wavfile)
            # feature_1D = wav2features_1D(wavfile)
            feature_mfcc = wav2features_mfcc_IICC(wavfile)#IICC
            feature_melspectrogram = wav2features_melspectrogram(wavfile)


            waves_vectors.append(feature_mfcc)
            waves_vectors_mel.append(feature_melspectrogram)

            label_array.append(index)

    ### compal data ::: wavs/1211_9505c0_onlyTest/9505c0/Sound/Painful/Painful_9505c0_2020-10-07_22-50-51.wav ###
    # for files in file_name:
    #     file_sort.append(((files.split('/')[-1]).split('_',2)[-1]).split('.')[0])
    
    ### unicharm data ::: wavs/1214_use_android_train5/test/a0981786817@gmail.com/Sound/Pain/1803071302.wav ###
    # for files in file_name:
    #     file_sort.append(files.split('/')[-1])

    ### lungyo data ::: wavs/0102-lungyo-split/0102-split-to-Personal/72/Sound/Painful/P_2_72@494.wav ###
    for files in file_name:
        file_sort.append(re.split('_|@|\.',files.split('/')[-1])[3])


    #### 樣本不能是0 ??
    X = [x for _ ,x in sorted(zip(file_sort,waves_vectors))]
    X1 = [x for _ ,x in sorted(zip(file_sort,waves_vectors_mel))]

    Y = [x for _ ,x in sorted(zip(file_sort,label_array))]

    X = np.array(X)
    X1 = np.array(X1)
    X1 = X1.reshape(-1,79,128)
    Y = to_categorical(Y)

    return X,Y,wav_num,X1

def get_testWav_personal(path):
    
    X,Y,wav_num,X1 = load_data_to_array_personal(path)

    return X, Y, wav_num,X1

def load_data_to_array_with_other_feature(path):
    labels, _, label_oneHot = get_labels(path)
    waves_vectors = []
    waves_vectors_zcr = []

    label_array = []
    wav_num = []
    wav_num.append(0)

    file_name = []
    file_sort = []

    for index, label in enumerate(labels):
        wavfiles = [ os.path.join(path,label,wavfile) for wavfile in os.listdir(path + '/' + label)]
        
        wavfiles = sorted(wavfiles)
        wav_num.append(len(wavfiles))
        for wavfile in tqdm(wavfiles, "Processing of label - '{}'".format(label)):
            file_name.append(wavfile)

            feature_1D = wav2features_1D(wavfile)
            feature_melspectrogram = wav2features_melspectrogram(wavfile)
            feature_zcr = wav2features_zcr(wavfile)

            waves_vectors.append(feature_1D)
            # waves_vectors_zcr.append(feature_melspectrogram)
            waves_vectors_zcr.append(feature_zcr)


            label_array.append(index)

    X = np.array(waves_vectors)
    X1 = np.array(waves_vectors_zcr) 
    Y = to_categorical(label_array)

    return X,Y,wav_num,X1

def get_train_test_with_other_feature(path):

    split_ratio=0.8
    random_state=42
    # X,Y,_,X1= load_data_to_array_with_other_feature(path)
    X,Y,_,X1= load_data_to_array_with_other_feature_data_augme(path)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= (1 - split_ratio), shuffle=True, stratify=Y, random_state=random_state)
    x1_train, x1_test, _, _ = train_test_split(X1, Y, test_size= (1 - split_ratio), shuffle=True, stratify=Y, random_state=random_state)


    return x_train, x_test, y_train, y_test,x1_train, x1_test

def get_testWav_with_other_feature(path):
    
    # X,Y,wav_num,X1= load_data_to_array_with_other_feature(path)
    X,Y,wav_num,X1= load_data_to_array_with_other_feature_data_augme(path)

    return X, Y, wav_num,X1

def load_data_to_array_with_other_feature_data_augme(path):
    labels, _, label_oneHot = get_labels(path)
    waves_vectors = []
    waves_vectors_zcr = []

    label_array = []
    wav_num = []
    wav_num.append(0)

    file_name = []
    file_sort = []

    for index, label in enumerate(labels):
        wavfiles = [ os.path.join(path,label,wavfile) for wavfile in os.listdir(path + '/' + label)]
        
        wavfiles = sorted(wavfiles)
        # random.shuffle(wavfiles)
        # wavfiles = wavfiles[:3500]
        wav_num.append(len(wavfiles))
        for wavfile in tqdm(wavfiles, "Processing of label - '{}'".format(label)):
            file_name.append(wavfile)

            feature_1D,feature_scale,feature_noise = wav2features_1D_data_augme(wavfile)
            feature_zcr = wav2features_zcr(wavfile)
            feature_melspectrogram = wav2features_melspectrogram(wavfile)

            waves_vectors.append(feature_1D)
            waves_vectors.append(feature_scale)
            waves_vectors.append(feature_noise)

            # waves_vectors_zcr.append(feature_zcr)
            # waves_vectors_zcr.append(feature_zcr)
            # waves_vectors_zcr.append(feature_zcr)

            waves_vectors_zcr.append(feature_melspectrogram)
            waves_vectors_zcr.append(feature_melspectrogram)
            waves_vectors_zcr.append(feature_melspectrogram)
            

            label_array.append(index)
            label_array.append(index)
            label_array.append(index)

    X = np.array(waves_vectors)
    X1 = np.array(waves_vectors_zcr) 
    Y = to_categorical(label_array)

    return X,Y,wav_num,X1

def wav2features_1D_data_augme(file_path):
    y, sr = librosa.load(file_path, mono=True, sr=None)
    y = librosa.util.normalize(y)

    noise = np.random.uniform(-1, 1, size=(len(y)))
    y_noise = y + noise

    y_scaled = (y/np.max(np.abs(y)))*0.5

    if(len(y)<40000):
        y = np.pad(y,(0,40000-len(y)),'mean')
        y_scaled = np.pad(y_scaled,(0,40000-len(y_scaled)),'mean')
        y_noise = np.pad(y_noise,(0,40000-len(y_noise)),'mean')

    else:
        y = y[:40000]
        y_scaled = y_scaled[:40000]
        y_noise = y_noise[:40000]

    return y,y_scaled,y_noise

def wav2features_mfcc_IICC_augme(file_path):
    y, sr = librosa.load(file_path, mono=True, sr=None)
    y = librosa.util.normalize(y)

    noise = np.random.uniform(-1, 1, size=(len(y)))
    y_noise = y + noise

    y_scaled = (y/np.max(np.abs(y)))*0.5


    if(len(y)<40000):
        y = np.pad(y,(0,40000-len(y)),'mean')
        y_scaled = np.pad(y_scaled,(0,40000-len(y_scaled)),'mean')
        y_noise = np.pad(y_noise,(0,40000-len(y_noise)),'mean')
    else:
        y = y[:40000]
        y_scaled = y_scaled[:40000]
        y_noise = y_noise[:40000]


    y_mfcc = librosa.feature.mfcc(y=y, sr=sr)
    y_scaled_mfcc = librosa.feature.mfcc(y=y_scaled, sr=sr)
    y_noise_mfcc = librosa.feature.mfcc(y=y_noise, sr=sr)

    return y_mfcc,y_scaled_mfcc,y_noise_mfcc

def load_data_to_array_data_augme(path):
    labels, _, label_oneHot = get_labels(path)
    waves_vectors = []

    label_array = []
    wav_num = []
    wav_num.append(0)

    file_name = []
    file_sort = []

    for index, label in enumerate(labels):
        wavfiles = [ os.path.join(path,label,wavfile) for wavfile in os.listdir(path + '/' + label)]
        
        wavfiles = sorted(wavfiles)
        wav_num.append(len(wavfiles))
        for wavfile in tqdm(wavfiles, "Processing of label - '{}'".format(label)):
            file_name.append(wavfile)

            # feature,feature_scale,feature_noise = wav2features_1D_data_augme(wavfile)
            feature,feature_scale,feature_noise = wav2features_mfcc_IICC_augme(wavfile)#IICC

            waves_vectors.append(feature)
            waves_vectors.append(feature_scale)
            waves_vectors.append(feature_noise)

            label_array.append(index)
            label_array.append(index)
            label_array.append(index)

    X = np.array(waves_vectors)
    Y = to_categorical(label_array)

    return X,Y,wav_num

def get_train_test_data_augme(path):

    split_ratio=0.8
    random_state=42
    X,Y,_ = load_data_to_array_data_augme(path)

    return train_test_split(X, Y, test_size= (1 - split_ratio), shuffle=True, stratify=Y, random_state=random_state)

def get_testWav_data_augme(path):
    
    X,Y,wav_num = load_data_to_array_data_augme(path)

    return X, Y, wav_num

def load_data_to_array_personal_with_otherFeat(path):
    labels, _, label_oneHot = get_labels(path)
    waves_vectors = []
    waves_vectors_mel = []

    waves_vectors_otherFeat = []

    label_array = []
    wav_num = []
    wav_num.append(0)

    file_name = []
    file_sort = []

    for index, label in enumerate(labels):
        wavfiles = [ os.path.join(path,label,wavfile) for wavfile in os.listdir(path + '/' + label)]
        
        wavfiles = sorted(wavfiles)
        wav_num.append(len(wavfiles))
        for wavfile in tqdm(wavfiles, "Processing of label - '{}'".format(label)):
            file_name.append(wavfile)
            feature = wav2features_1D(wavfile)
            # feature = wav2features_mfcc_IICC(wavfile)#IICC
            # feature_zcr = wav2features_zcr(wavfile)
            feature_otherFeat = wav2features_melspectrogram(wavfile)

            feature_melspectrogram = wav2features_melspectrogram(wavfile)


            waves_vectors.append(feature)
            waves_vectors_mel.append(feature_melspectrogram)

            waves_vectors_otherFeat.append(feature_otherFeat)

            label_array.append(index)

    ### compal data ::: wavs/1211_9505c0_onlyTest/9505c0/Sound/Painful/Painful_9505c0_2020-10-07_22-50-51.wav ###
    for files in file_name:
        file_sort.append(((files.split('/')[-1]).split('_',2)[-1]).split('.')[0])
    
    ### unicharm data ::: wavs/1214_use_android_train5/test/a0981786817@gmail.com/Sound/Pain/1803071302.wav ###
    # for files in file_name:
    #     file_sort.append(files.split('/')[-1])

    ### lungyo data ::: wavs/0102-lungyo-split/0102-split-to-Personal/72/Sound/Painful/P_2_72@494.wav ###
    # for files in file_name:
    #     file_sort.append(re.split('_|@|\.',files.split('/')[-1])[3])


    #### 樣本不能是0 ??
    X = [x for _ ,x in sorted(zip(file_sort,waves_vectors))]
    X1 = [x for _ ,x in sorted(zip(file_sort,waves_vectors_mel))]
    X2 = [x for _ ,x in sorted(zip(file_sort,waves_vectors_otherFeat))]

    Y = [x for _ ,x in sorted(zip(file_sort,label_array))]

    X = np.array(X)
    X1 = np.array(X1)
    X2 = np.array(X2)


    X1 = X1.reshape(-1,79,128)
    Y = to_categorical(Y)

    return X,Y,wav_num,X1,X2

def get_testWav_personal_with_otherFeat(path):
    
    X,Y,wav_num,X1,X2 = load_data_to_array_personal_with_otherFeat(path)

    return X, Y, wav_num,X1,X2


file_path = "./wavs/General/Hungry/Hungry_30064.wav"

print(wav2features_melspectrogram(file_path)) 




























