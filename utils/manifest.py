import soundfile as sf
import numpy as np
import librosa


def _calc_alpha(SNR, speech, noise):
    alpha = np.sqrt(np.sum(speech ** 2.0) / (np.sum(noise ** 2.0) * (10.0 ** (SNR / 10.0))))
    return alpha


def _calc_stft(data):
    feature = librosa.stft(data, n_fft=320, win_length=320, hop_length=160)
    return feature


def _calc_irm(speech, noise):
    _s_square = np.square(np.absolute(speech))
    _n_square = np.square(np.absolute(noise))
    irm_mask = np.sqrt(np.divide(_s_square, (_s_square + _n_square)))
    return irm_mask


def _calc_cirm(speech, mix):
    speech_real = np.real(speech)
    speech_imag = np.imag(speech)

    mix_real = np.real(mix)
    mix_imag = np.imag(mix)

    mask_real = np.divide(mix_real * speech_real + mix_imag * speech_imag, np.square(mix_real) + np.square(mix_imag))
    mask_imag = np.divide(mix_real * speech_imag - mix_imag * speech_real, np.square(mix_real) + np.square(mix_imag))

    return mask_real, mask_imag


if __name__ == '__main__':
    speech_wav = '/home/zhangpeng/se_temp/clean.wav'
    noise_file_wav = '/home/zhangpeng/se_temp/factory.wav'
    noise_wav = '/home/zhangpeng/se_temp/noise.wav'
    mix_wav = '/home/zhangpeng/se_temp/mix.wav'
    est_wav = '/home/zhangpeng/se_temp/est.wav'

    # 1. 读取纯净语音和噪音
    speech, _ = sf.read(speech_wav)
    noise, _ = sf.read(noise_file_wav)

    # 2. 因为噪声比较长，所以从噪声文件中随机截取和纯净语音一样长的噪声
    speech_len = len(speech)
    noise_len = len(noise)
    max_idx = noise_len - speech_len
    idx = np.random.randint(low=0, high=max_idx)
    noise = noise[idx:idx + speech_len]
    assert len(noise) == speech_len

    # 3. 加0dB的噪声，生成加噪语音
    mix_0dB = speech + noise * _calc_alpha(SNR=0, speech=speech, noise=noise)
    noise_0dB = noise * _calc_alpha(SNR=0, speech=speech, noise=noise)
    sf.write(file=mix_wav, data=mix_0dB, samplerate=16000)
    sf.write(file=noise_wav, data=noise_0dB, samplerate=16000)

    # 4. 提取纯净语音speech，noise, mix的频谱特征，
    speech_spec = _calc_stft(speech)
    noise_spec = _calc_stft(noise)
    mix_spec = _calc_stft(mix_0dB)

    # 5. 计算IRM
    irm_mask = _calc_irm(speech=speech_spec, noise=noise_spec)

    # 6.乘到加噪语音上， 得到降噪后的语音
    est = mix_spec * irm_mask

    # 7. 合成降噪后的语音
    est_pcm = librosa.istft(est, win_length=320, hop_length=160)
    sf.write(file=est_wav, data=est_pcm, samplerate=16000)

