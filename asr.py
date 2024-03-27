"""Modal demo for Gazelle as an 'ASR' service.

Cascaded voice systems take the speech -> ASR -> LLM -> TTS.
Because Gazelle combines an audio encoder with language model, we do not need to use a separate ASR system.
Gazelle takes in audio and directly outputs a textual response. As a result, it can be directly plugged into an existing
cascaded system as a transcriber. Here, we setup a very basic demo to do so.
"""

import time

import modal
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

stub = modal.Stub("gazelle-asr-demo")

MODEL_NAME = "tincans-ai/gazelle-v0.2"
AUDIO_MODEL_NAME = "facebook/wav2vec2-base-960h"
MODEL_DIR = "/model"


def download_model():
    import os

    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_DIR,
    )
    move_cache()


gazelle_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.2.1",
        "transformers==4.38.2",
        "git+https://github.com/tincans-ai/gazelle@main",
        "hf-transfer",
    )
    .env(
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    )
    .run_function(
        download_model,
        secrets=[modal.Secret.from_name("hf_read_token")],
        timeout=60 * 20,
    )
)

with gazelle_image.imports():
    from threading import Thread

    import numpy as np
    import torch
    import torchaudio
    from gazelle import GazelleConfig, GazelleForConditionalGeneration
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        TextIteratorStreamer,
    )


@stub.cls(
    image=gazelle_image,
    gpu="A10G",
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("hf_read_token")],
    # not production ready, just for a single test!
    concurrency_limit=1,
)
class GazelleModel:
    @modal.enter()
    def load_model(self):
        t0 = time.time()
        print("Loading model...")

        config = GazelleConfig.from_pretrained(MODEL_NAME)

        self.model = GazelleForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            config=config,
            torch_dtype=torch.bfloat16,
        )

        print(f"Model loaded in {time.time() - t0:.2f}s")

        self.audio_processor = AutoProcessor.from_pretrained(AUDIO_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self.model.config.use_cache = True

        self.model.cuda()
        self.model.eval()

    def prewarm(self):
        """Prewarm the model by generating a dummy response.

        Call this method to prewarm the model before serving requests. Wait for response before returning.
        """
        print("prewarming!")
        return True

    @modal.method()
    async def generate(self, input="", audio=None, history=[]):
        if input == "" and not audio:
            return

        if "<|audio|>" in input and not audio:
            raise ValueError(
                "Audio input required if '<|audio|>' token is present in input"
            )

        if audio and "<|audio|>" not in input:
            input = "<|audio|> \n\n" + input

        t0 = time.time()

        assert len(history) % 2 == 0, "History must be an even number of messages"

        if audio:
            sr, audio_data = audio
            if audio_data.dtype == "int16":
                audio_data_float = audio_data.astype(np.float32) / 32768.0
                audio_data = torch.from_numpy(audio_data_float)
            elif audio_data.dtype == "int32":
                audio_data_float = audio_data.astype(np.float32) / 2147483648.0
                audio_data = torch.from_numpy(audio_data_float)
            else:
                audio_data = torch.from_numpy(audio_data)

            if sr != 16000:
                # resample
                print("Resampling audio from {} to 16000".format(sr))
                audio_data = torchaudio.transforms.Resample(sr, 16000)(audio_data)
            # print(audio_data)
            print(audio_data.shape)
            audio_values = self.audio_processor(
                audio=audio_data, sampling_rate=16000, return_tensors="pt"
            ).input_values
            audio_values = audio_values.to(dtype=torch.bfloat16, device="cuda")

        messages = []
        for i in range(0, len(history), 2):
            messages.append({"role": "user", "content": history[i]})
            messages.append({"role": "user", "content": history[i + 1]})

        messages.append({"role": "user", "content": input})
        print(messages)
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).cuda()

        generation_kwargs = dict(
            inputs=tokenized_chat,
            audio_values=audio_values if audio else None,
            streamer=self.streamer,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.2,
            max_new_tokens=256,
        )

        # Run generation on separate thread to enable response streaming.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        results = []
        first_token_time = None
        for new_text in self.streamer:
            yield new_text
            if not first_token_time:
                first_token_time = time.time()
            results.append(new_text)
        thread.join()

        end_time = time.time()
        ttft = first_token_time - t0
        total_time = end_time - t0
        print(f"Output generated. TTFT: {ttft:.2f}s, Total: {total_time:.2f}s")


# @stub.local_entrypoint()
@stub.function()
@modal.web_endpoint(method="POST")
def stream(args: dict):
    model = GazelleModel()

    resp = model.generate.remote_gen(args.get("input", "<|audio|>"), audio=args.get("audio"), history=args.get("history", []))

    return StreamingResponse(
        resp,
        media_type="text/event-stream",
    )
