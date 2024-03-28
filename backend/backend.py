import base64
import dataclasses
import traceback
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils


# Taken from utils_vad.py
def validate(model, inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs


# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


class AudioEncoding(str, Enum):
    linear16: str = "linear16"
    mulaw: str = "mulaw"


class WebSocketMessage(BaseModel):
    type: str


class start_message(WebSocketMessage):
    type: str = "websocket_start"
    # transcriber_config: TranscriberConfig
    # agent_config: AgentConfig
    # synthesizer_config: SynthesizerConfig
    conversation_id: Optional[str] = None
    first_message: Optional[str] = None
    system_prompt: Optional[str] = None


class AudioMessage(WebSocketMessage):
    type: str = "websocket_audio"
    data: str


class ReadyMessage(WebSocketMessage):
    type: str = "websocket_ready"


class StopMessage(WebSocketMessage):
    type: str = "websocket_stop"


class InputAudioConfig(BaseModel):
    sampling_rate: int
    audio_encoding: AudioEncoding
    chunk_size: int
    downsampling: Optional[int] = None


class OutputAudioConfig(BaseModel):
    sampling_rate: int
    audio_encoding: AudioEncoding


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AudioConfigStartMessage(WebSocketMessage):
    type: str = "websocket_audio_config_start"
    input_audio_config: InputAudioConfig
    output_audio_config: OutputAudioConfig
    conversation_id: Optional[str] = None
    subscribe_transcript: Optional[bool] = None
    first_message: Optional[str] = None
    system_prompt: Optional[str] = None


@dataclasses.dataclass
class Handler:
    input_audio_config: InputAudioConfig = None
    output_audio_config: OutputAudioConfig = None
    conversation_id: Optional[str] = None
    subscribe_transcript: Optional[bool] = None
    first_message: Optional[str] = None
    system_prompt: Optional[str] = None


class VADState:
    def __init__(self):
        self.vad = None


MIN_SPEECH_DURATION_MS = 250
SPEECH_PAD_MS = 30
MIN_SILENCE_DURATION_MS = 100
SAMPLING_RATE = 16_000
WINDOW_SIZE_SAMPLES = 512


def get_vad_state():
    return VADState()


class VADNotInitializedException(Exception):
    pass


async def reset_model_states(websocket: WebSocket, vad_state: VADState):
    if vad_state.vad is None:
        raise VADNotInitializedException()
    vad_state.vad.reset_states()
    await websocket.send_json(
        {
            "message_type": "model_states_reset",
            "message": "Model state successfully reset",
        }  # noqa: E501
    )


@app.websocket("/conversation")
async def conversation(websocket: WebSocket):
    await websocket.accept()
    raw_audio_buffer = b""
    audio_buffer = torch.tensor([], dtype=torch.float32)
    speech_probs = []
    buffer = torch.tensor([], dtype=torch.float32)

    handler = Handler()
    vad_state = VADState()
    resample_transform = None
    while True:
        try:
            message = await websocket.receive_json()
            message_type = message["type"]
            # print(f"Received message: {message_type}", message)

            if message_type == "websocket_audio_config_start":
                config_message = AudioConfigStartMessage(**message)
                handler = Handler(
                    input_audio_config=config_message.input_audio_config,
                    output_audio_config=config_message.output_audio_config,
                    conversation_id=config_message.conversation_id,
                    subscribe_transcript=config_message.subscribe_transcript,
                    first_message=config_message.first_message,
                    system_prompt=config_message.system_prompt,
                )

                kwargs = {
                    "lowpass_filter_width": 16,
                    "rolloff": 0.85,
                    "resampling_method": "sinc_interp_kaiser",
                    "beta": 8.555504641634386,
                }
                resample_transform = torchaudio.transforms.Resample(
                    orig_freq=handler.input_audio_config.sampling_rate,
                    new_freq=SAMPLING_RATE,
                    **kwargs,
                )

                vad_state.vad = VADIterator(
                    model=vad_model,
                    threshold=0.3,
                    # always use 16k sampling rate
                    sampling_rate=16000,
                    min_silence_duration_ms=(MIN_SILENCE_DURATION_MS),
                    speech_pad_ms=(SPEECH_PAD_MS),
                )

                # return ready message
                print("Sending ready message")
                await websocket.send_json(ReadyMessage().model_dump())
            elif message_type == "websocket_audio":
                audio_message = AudioMessage(**message)
                audio_data = base64.b64decode(audio_message.data)
                
                raw_audio_buffer += audio_data

                waveform, _ = torchaudio.load(BytesIO(raw_audio_buffer))

                # resample waveform to 16k
                if handler.input_audio_config.sampling_rate != SAMPLING_RATE:
                    waveform = resample_transform(waveform)

                # Update the buffer
                buffer = torch.cat((buffer, waveform.squeeze()))

                # First, do VAD on the most recent audio buffer
                # Conditionally process audio
                num_samples = len(buffer)
                if num_samples >= WINDOW_SIZE_SAMPLES:
                    remainder = num_samples % WINDOW_SIZE_SAMPLES
                    split_index = num_samples - remainder
                    # Split buffer into audio and remainder
                    audio, buffer = (
                        buffer[..., :split_index],
                        buffer[..., split_index:],
                    )  # noqa: E501
                    audio_buffer = torch.cat((audio_buffer, audio))
                    # Get speech timestamps
                    vad_res = vad_state.vad(audio, return_seconds=False)  # noqa: E501
                    if vad_res is not None:
                        print(vad_res)


                # TODO: Perform further processing on the audio buffer
            else:
                raise ValueError("Invalid message type")

        except Exception as e:
            print(traceback.format_exc())
            break

    # TODO: Perform final processing on the complete audio buffer
    await websocket.close()
