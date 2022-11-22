import scipy.io.wavfile
import numpy as np
import sys

from   agc import tf_agc


# read audiofile
original_file_name = "speech.wav"
MASTER_WORK_DIR = "./"
sr, d = scipy.io.wavfile.read(MASTER_WORK_DIR + original_file_name)

# convert from int16 to float (-1,1) range
convert_16_bit = float(2 ** 15)
d = d / (convert_16_bit + 1.0)

# apply AGC
(y, D, E) = tf_agc(d, sr)

# convert back to int16 to save
y = np.int16(y / np.max(np.abs(y)) * convert_16_bit)
scipy.io.wavfile.write(MASTER_WORK_DIR + 'speech_agc.wav', sr, y)
