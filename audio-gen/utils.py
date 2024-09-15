
from io import BytesIO
import soundfile
import numpy as np

def audio_array_to_buffer(
        audio_array: np.array,
        sample_rate,
        format = "wav",
) -> BytesIO:
    buffer = BytesIO()
    soundfile.write(buffer, audio_array, sample_rate, format=format)
    buffer.seek(0)
    return buffer