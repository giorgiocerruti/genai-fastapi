from transformers import AutoProcessor, AutoModel, BarkProcessor, BarkModel
from schemas import VoicePresets
import numpy as np

def load_audio_model() -> tuple[BarkProcessor, BarkModel]:
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small")

    return processor, model

def generate_audio(
        processor: BarkProcessor,
        model: BarkModel,
        prompt: str,
        preset: VoicePresets,
) -> tuple[np.array, int]:
    inputs = processor(text=[prompt], return_tensors="pt", voice_preset=preset)
    output = model.generate(**inputs, do_sample=True).cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    return output, sample_rate
