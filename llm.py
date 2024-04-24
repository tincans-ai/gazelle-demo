"""Modal + Gradio demo of Gazelle.

This demo uses Modal to serve a Gradio app that interfaces with Gazelle, a joint speech-language model by Tincans.

We utilize two separate images, one with GPU and ML frameworks, and another with just web dependencies.
The web image is used to serve the Gradio app, while the GPU image is used to serve the model.

Because Modal uses serverless GPU's, it is cheaper to serve over a long period of time if there is not sustained demand.
In periods of high demand, it can be more expensive than reserved capacity, but can scale out faster to improve throughput.
"""

import time

import modal
from fastapi import FastAPI

stub = modal.Stub("gazelle-demo")

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
    container_idle_timeout=120,
    secrets=[modal.Secret.from_name("hf_read_token")],
    concurrency_limit=32,
    keep_warm=1,
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

        ttft = time.time() - first_token_time
        total_time = time.time() - t0
        print(f"Output generated. TTFT: {ttft:.2f}s, Total: {total_time:.2f}s")


@stub.local_entrypoint()
def main(input: str):
    model = GazelleModel()
    for val in model.generate.remote_gen(input):
        print(val, end="", flush=True)


def download_samples():
    import concurrent.futures
    import os

    import requests

    remote_urls = [
        "test6.wav",
        "test21.wav",
        "test26.wav",
        "testnvidia.wav",
        "testdoc.wav",
        "testappt3.wav",
    ]

    def download_file(url):
        base_url = "https://r2proxy.tincans.ai/"
        full_url = base_url + url
        filename = os.path.basename(url)
        if not os.path.exists(filename):
            response = requests.get(full_url)
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {filename}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(download_file, remote_urls)


# Gradio frontend logic

web_app = FastAPI()

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi==0.110.0", "gradio==3.50.2")
    .run_function(download_samples)
)


@stub.cls(
    image=web_image,
    concurrency_limit=128,
    container_idle_timeout=300,
    keep_warm=2,
)
class GradioWrapper:
    def __init__(self):
        self.app_ = web_app

    @modal.enter()
    def startup(self):
        import os

        import gradio as gr
        from gradio.routes import mount_gradio_app

        gz = GazelleModel()

        def gen_(input, mic_audio, upload_audio):
            final_str = ""
            audio = None
            if mic_audio:
                audio = mic_audio
            elif upload_audio:
                audio = upload_audio
            if mic_audio and upload_audio:
                raise ValueError("Only one audio input is allowed")

            print("about to call remote gen")
            for result in gz.generate.remote_gen(input, audio):
                final_str += result
                yield final_str

        examples = [
            ["", None, os.path.join(os.path.dirname(__file__), "test6.wav")],
            ["", None, os.path.join(os.path.dirname(__file__), "test26.wav")],
            [
                "You are a professional with no available time slots for the rest of the week.",
                None,
                os.path.join(os.path.dirname(__file__), "testappt3.wav"),
            ],
            [
                "You are an expert diagnostic doctor.",
                None,
                os.path.join(os.path.dirname(__file__), "testdoc.wav"),
            ],
            [
                "Translate the previous statement to French.",
                None,
                os.path.join(os.path.dirname(__file__), "test6.wav"),
            ],
            [
                "Why would the Chinese government increase social spending?",
                None,
                os.path.join(os.path.dirname(__file__), "test21.wav"),
            ],
            [
                "What is Nvidia's new generation of chips called? When will they ship?",
                None,
                os.path.join(os.path.dirname(__file__), "testnvidia.wav"),
            ],
            [
                "Translate the previous statement to Chinese.",
                None,
                os.path.join(os.path.dirname(__file__), "testnvidia.wav"),
            ],
        ]

        gr_theme = gr.themes.Default(
            font=[gr.themes.GoogleFont("Space Grotesk"), "Arial", "sans-serif"]
        )

        interface = gr.Interface(
            fn=gen_,
            # fn=gen_dummy,
            theme=gr_theme,
            inputs=[
                "textbox",
                gr.Audio(source="microphone"),
                gr.Audio(source="upload"),
            ],
            outputs="textbox",
            title="🦌 Gazelle v0.2",
            description="""Gazelle is a joint speech-language model by [Tincans](https://tincans.ai) 🥫 - for more details and prompt ideas, see our [v0.2 announcement](https://tincans.ai/slm3). This is an *early research preview* -- please temper expectations!
        Gazelle can take in text and audio as input (interchangeably) and generates text as output.
        You can further synthesize the text output into audio via a TTS provider (not implemented here). Some example tasks include transcribing audio, answering questions, or understanding spoken audio. This approach will be superior for business use cases where latency and conversational quality matter - such as customer support, outbound sales, and more.
        
        Known limitations exist! The model was only trained on English audio and is not expected to work well with other languages. Similarly, the model does not handle accents well yet. The gradio demo may have bugs with sample rate for audio. We also only accept a single audio input (microphone or upload).

        Inference is done via serverless GPU's on [Modal](https://modal.com). As such, you may experience cold start delays (about 30 seconds) on first use, but subsequent responses will be faster.
        This demo is purposefully not optimized for inference speed, but rather to showcase the capabilities of Gazelle. We do not store any responses.

        Feedback? [Twitter](https://twitter.com/hingeloss) | [email](hello@tincans.ai) | [GitHub](https://github.com/tincans-ai/gazelle)
        """,
            examples=examples,
        )

        interface.queue()
        interface.startup_events()
        self.blocks = interface

        self.app_ = mount_gradio_app(
            app=web_app,
            blocks=interface,
            path="/",
        )

    @modal.exit()
    def exit(self):
        self.blocks.close()
        print("shutdown Gradio blocks")

    @modal.asgi_app(label="gazelle-demo", custom_domains=["demo.tincans.ai"])
    def app(self):
        return self.app_
