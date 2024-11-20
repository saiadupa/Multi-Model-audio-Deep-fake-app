import os
from pydub import AudioSegment
import soundfile as sf

def convert_to_wav(src, dst, st='PCM_32'):

    print("Writing {} to {}".format(src, dst))
    data, samplerate = sf.read(src)

    # print(data.dtype)
    # print(data)
    # print(np.min(data))
    # print(np.max(data))

    sf.write(dst, data, samplerate, subtype=st)

def convert_mp3_to_wav(src, dst):

    # convert wav to mp3                                                            
    sound = AudioSegment.from_mp3(src)
    # sound = AudioSegment.from_file(file = src, format = "wav")
    sound = sound.set_channels(1)
    sound.export(dst, format="wav")


if __name__ == "__main__":

    data_dir = './data'
    in_filename = 'Romanian_President_2.wav'
    out_filename = in_filename.split('.')[0] + '.wav'

    in_file = os.path.join(data_dir, in_filename)
    out_file = os.path.join(data_dir, out_filename)

    convert_mp3_to_wav(in_file, out_file)
    convert_to_wav(out_file, out_file, st='PCM_16')