"""Modal + Gradio demo of Gazelle.

This demo uses Modal to serve a Gradio app that interfaces with Gazelle, a joint speech-language model by Tincans.

We utilize two separate images, one with GPU and ML frameworks, and another with just web dependencies.
The web image is used to serve the Gradio app, while the GPU image is used to serve the model.

Because Modal uses serverless GPU's, it is cheaper to serve over a long period of time if there is not sustained demand.
In periods of high demand, it can be more expensive than reserved capacity, but can scale out faster to improve throughput.
"""


import shlex
import subprocess
import modal

from pathlib import Path

app = modal.App("gazelle-demo")



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


web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi==0.110.0", "gradio==3.50.2")
    .run_function(download_samples)
)

gradio_script_local_path = Path(__file__).parent / "gradio_app.py"
gradio_script_remote_path = Path("/root/gradio_app.py")

if not gradio_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

gradio_script_mount = modal.Mount.from_local_file(
    gradio_script_local_path,
    gradio_script_remote_path,
)


@app.function(image=web_image, mounts=[gradio_script_mount])
@modal.web_server(8000,custom_domains=["demo.tincans.ai"])
def web_app():
    target = shlex.quote(str(gradio_script_remote_path))
    cmd = f"python {target}"
    subprocess.Popen(cmd, shell=True)