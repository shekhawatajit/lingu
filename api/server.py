import io, os, json, base64, wave, asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
from dotenv import load_dotenv
from azure_voice_live import AsyncAzureVoiceLive  

AUDIO_SR = 24000
FRAME_MS = 20
SAMPLES_PER_FRAME = int(AUDIO_SR * (FRAME_MS/1000.0))
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2  # int16 mono

app = FastAPI()
load_dotenv()

def audio_to_pcm_24k_mono(raw: bytes, mime: str | None) -> bytes:
    # Let ffmpeg sniff; format=None works if the container has a header
    seg = AudioSegment.from_file(io.BytesIO(raw), format=None)
    seg = seg.set_channels(1).set_frame_rate(AUDIO_SR).set_sample_width(2)
    return seg.raw_data

def chunk_pcm_20ms(pcm: bytes) -> list[bytes]:
    if len(pcm) % BYTES_PER_FRAME != 0:
        pad = BYTES_PER_FRAME - (len(pcm) % BYTES_PER_FRAME)
        pcm += b"\x00" * pad
    return [pcm[i:i+BYTES_PER_FRAME] for i in range(0, len(pcm), BYTES_PER_FRAME)]

def pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_SR)
        wf.writeframes(pcm)
    return buf.getvalue()

async def stream_to_azure_and_back(frames: list[bytes]) -> bytes:
    client = AsyncAzureVoiceLive(
        azure_endpoint=os.environ["AZURE_VOICE_LIVE_ENDPOINT"],
        api_key=os.environ.get("AZURE_VOICE_LIVE_API_KEY"),
    )
    collected = bytearray()
    async with client.connect(model=os.environ["AZURE_VOICE_LIVE_DEPLOYMENT"]) as conn:
        await conn.session.update(session={
            "turn_detection": {
                "type": "azure_semantic_vad",
                "threshold": 0.3,
                "prefix_padding_ms": 200,
                "silence_duration_ms": 500,
                "remove_filler_words": False,
                "end_of_utterance_detection": { "model": "semantic_detection_v1", "threshold": 0.01, "timeout": 2 }
            },
            "input_audio_noise_reduction": {"type": "azure_deep_noise_suppression"},
            "input_audio_echo_cancellation": {"type": "server_echo_cancellation"},
            "voice": { "name": "en-US-Ava:DragonHDLatestNeural", "type": "azure-standard", "temperature": 0.8 },
        })

        for frame in frames:
            await conn.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode("utf-8"),
                "event_id": ""
            }))

        await conn.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await conn.send(json.dumps({"type": "response.create"}))

        async for raw in conn:
            evt = json.loads(raw)
            if evt.get("type") == "response.audio.delta":
                delta = evt.get("delta")
                if delta:
                    collected.extend(base64.b64decode(delta))
            elif evt.get("type") == "response.done":
                break

    return bytes(collected)

@app.post("/s2s")
async def s2s(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        pcm = audio_to_pcm_24k_mono(raw, file.content_type)
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio: {e}")

    frames = chunk_pcm_20ms(pcm)
    reply_pcm = await stream_to_azure_and_back(frames)
    reply_wav = pcm_to_wav(reply_pcm)
    return StreamingResponse(io.BytesIO(reply_wav), media_type="audio/wav")
