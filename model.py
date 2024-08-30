import time
import modal

AUDIO_TOKEN = "<|audio|>"
MODEL_NAME = "tincans-ai/gazelle-v0.2"
AUDIO_MODEL_NAME = "facebook/wav2vec2-base-960h"
MODEL_DIR = "/model"


app = modal.App("gazelle-demo-model")


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
        "numpy<2",
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
        AutoFeatureExtractor,
        TextIteratorStreamer,
    )


@app.cls(
    image=gazelle_image,
    gpu="A10G",
    container_idle_timeout=120,
    secrets=[modal.Secret.from_name("hf_read_token")],
    concurrency_limit=8,
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

        if "bert" in AUDIO_MODEL_NAME:
            self.audio_processor = AutoFeatureExtractor.from_pretrained(
                AUDIO_MODEL_NAME
            )
        else:
            self.audio_processor = AutoProcessor.from_pretrained(AUDIO_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self.model.config.use_cache = True
        self.model.cuda()
        self.model.eval()

    @modal.method()
    async def generate(self, input="", audio=None, history=[]):
        if input == "" and not audio:
            return

        if AUDIO_TOKEN in input and not audio:
            raise ValueError(
                f"Audio input required if '{AUDIO_TOKEN}' token is present in input"
            )

        if audio and AUDIO_TOKEN not in input:
            input = f"{AUDIO_TOKEN} \n\n" + input

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

        ttft = time.time() - first_token_time
        total_time = time.time() - t0
        print(f"Output generated. TTFT: {ttft:.2f}s, Total: {total_time:.2f}s")


@app.local_entrypoint()
def main(input: str):
    model = GazelleModel()
    for val in model.generate.remote_gen(input):
        print(val, end="", flush=True)