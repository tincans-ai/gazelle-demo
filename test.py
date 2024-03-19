import time
import modal
from fastapi import FastAPI

stub = modal.Stub("gradio-repo")

web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi", "gradio==4.1.0"
)

web_app = FastAPI()

@stub.cls(
    image=web_image,
)
class AppClass:
    def __init__(self):
        self.app_ = web_app

    @modal.enter()
    def startup(self):
        import gradio as gr
        from gradio.routes import mount_gradio_app
        import contextlib

        def fake_llm_text_generator(message):
            final_str = ""
            for result in ["hello!", " world!", " this", " is", " a", " test"]:
                time.sleep(0.3)
                final_str += result
                yield final_str

            print("done", final_str)

        demo = gr.Interface(
            fn=fake_llm_text_generator,
            inputs="textbox",
            outputs="textbox",
            title="Test",
            description="This is a test",
        )

        demo.queue()
        old_lifespan = web_app.router.lifespan_context

        demo.startup_events()
        @contextlib.asynccontextmanager
        async def new_lifespan(app: FastAPI):
            async with old_lifespan(
                app
            ):  # Instert the startup events inside the FastAPI context manager
                demo.startup_events()
                yield

        web_app.router.lifespan_context = new_lifespan
        self.blocks = demo
        self.app_ = mount_gradio_app(
            app=web_app,
            blocks=demo,
            path="/",
        )

    @modal.exit()
    def exit(self):
        self.blocks.close()
        print("shutdown Gradio blocks")

    @modal.asgi_app()
    def app(self):
        return self.app_