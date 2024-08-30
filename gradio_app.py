import os
import modal
from fastapi import FastAPI
import gradio as gr
from gradio.routes import mount_gradio_app


gz = modal.Cls.lookup("gazelle-demo-model", "GazelleModel")


def gen_(input, mic_audio, upload_audio):
    final_str = ""
    audio = None
    if mic_audio:
        audio = mic_audio
    elif upload_audio:
        audio = upload_audio
    if mic_audio and upload_audio:
        raise ValueError("Only one audio input is allowed")

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
    theme=gr_theme,
    inputs=[
        "textbox",
        gr.Audio(source="microphone"),
        gr.Audio(source="upload"),
    ],
    outputs="textbox",
    title="🦌 Gazelle",
    description="""Gazelle is a joint speech-language model by [Tincans](https://tincans.ai) 🥫.

For more details and prompt ideas, see our [v0.2 announcement](https://tincans.ai/slm3). This is an *early research preview* -- please temper expectations!
Gazelle can interchangeably take in text and audio as input and generates a response in text.
You can further synthesize the text output into audio via a TTS provider (not implemented here). Example tasks include transcribing audio, answering questions, or understanding spoken audio. This approach is superior for business use cases where latency and conversational quality matter - such as customer support, outbound sales, and more.

Inference is done via serverless GPU's on [Modal](https://modal.com). As such, you may experience cold start delays (about 30 seconds) on first use, but subsequent responses will be faster.
This demo is purposefully not optimized for inference speed, but rather to showcase the capabilities of Gazelle. We do not store any responses.

Feedback? [Twitter](https://twitter.com/hingeloss) | [email](hello@tincans.ai) | [GitHub](https://github.com/tincans-ai/gazelle)
""",
    examples=examples,
)

interface.queue(max_size=5)
interface.startup_events()

web_app = FastAPI()

app = mount_gradio_app(
    app=web_app,
    blocks=interface,
    path="/",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)