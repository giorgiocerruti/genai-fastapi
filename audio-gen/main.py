from fastapi import FastAPI, Query,  status
from fastapi.responses import StreamingResponse
from models import load_audio_model, generate_audio
from schemas import VoicePresets
from utils import audio_array_to_buffer

app = FastAPI()

@app.get(
    "/generate/audio",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse,
    )
def server_tex_to_audio_model_controller(
    prompt=Query(...),
    preset: VoicePresets = Query(default="v2/en_speaker_1"),
):
    processor, model = load_audio_model()
    output, sample_rate = generate_audio(processor=processor, model=model, prompt=prompt, preset=preset)
    return StreamingResponse(audio_array_to_buffer(output, sample_rate), media_type="audio/wav")


if __name__ == "__main__":
        import uvicorn
        uvicorn.run("main:app", port=8000, reload=True)