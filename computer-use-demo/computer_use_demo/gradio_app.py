"""
Entrypoint for Gradio, see https://www.gradio.app/
"""

import asyncio
import base64
import os
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import PosixPath
from typing import cast

import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock

from computer_use_demo.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    sampling_loop,
)
from computer_use_demo.tools import ToolResult

CONFIG_DIR = PosixPath("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

WARNING_TEXT = "⚠️ Security Alert: Never provide access to sensitive accounts or data, as malicious web content can hijack Claude's behavior"

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"

def load_from_storage(filename: str) -> str | None:
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        print(f"Debug: Error loading {filename}: {e}")
    return None

def save_to_storage(filename: str, data: str) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        file_path.chmod(0o600)
    except Exception as e:
        print(f"Debug: Error saving {filename}: {e}")

def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key to continue."
    # ... (other provider checks remain the same)

async def chat(message, history, api_key, provider, model, custom_system_prompt, only_n_most_recent_images, hide_images):
    history.append({"role": "user", "content": message})
    messages = [{"role": m["role"], "content": [TextBlock(type="text", text=m["content"])]} for m in history]

    async def output_callback(sender: Sender, content):
        if isinstance(content, str):
            yield content
        elif isinstance(content, (BetaTextBlock, TextBlock)):
            yield content.text
        elif isinstance(content, (BetaToolUseBlock, ToolUseBlock)):
            yield f"Tool Use: {content.name}\nInput: {content.input}"
        elif isinstance(content, ToolResult):
            if content.output:
                yield content.output
            if content.error:
                yield f"Error: {content.error}"
            if content.base64_image and not hide_images:
                image = gr.Image(value=base64.b64decode(content.base64_image))
                yield image

    bot_response = ""
    async for partial_response in sampling_loop(
        system_prompt_suffix=custom_system_prompt,
        model=model,
        provider=provider,
        messages=messages,
        output_callback=partial(output_callback, Sender.BOT),
        tool_output_callback=partial(output_callback, Sender.TOOL),
        api_key=api_key,
        only_n_most_recent_images=only_n_most_recent_images,
    ):
        bot_response += partial_response
        yield bot_response

    history.append({"role": "assistant", "content": bot_response})
    return history

def create_app():
    with gr.Blocks() as app:
        gr.Markdown("# Claude Computer Use Demo")
        gr.Markdown(WARNING_TEXT)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat")
                message = gr.Textbox(label="Type a message to send to Claude to control the computer...")
                send = gr.Button("Send")

            with gr.Column(scale=1):
                api_key = gr.Textbox(label="Anthropic API Key", type="password", value=load_from_storage("api_key") or os.getenv("ANTHROPIC_API_KEY", ""))
                provider = gr.Dropdown(label="API Provider", choices=[p.value for p in APIProvider], value=os.getenv("API_PROVIDER", "anthropic") or APIProvider.ANTHROPIC)
                model = gr.Textbox(label="Model", value=PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.ANTHROPIC])
                custom_system_prompt = gr.Textbox(label="Custom System Prompt Suffix", value=load_from_storage("system_prompt") or "", lines=3)
                only_n_most_recent_images = gr.Number(label="Only send N most recent images", value=10, minimum=0)
                hide_images = gr.Checkbox(label="Hide screenshots", value=False)
                reset = gr.Button("Reset")

        send.click(chat, inputs=[message, chatbot, api_key, provider, model, custom_system_prompt, only_n_most_recent_images, hide_images], outputs=chatbot)
        reset.click(lambda: None, outputs=[chatbot], queue=False)

        api_key.change(lambda x: save_to_storage("api_key", x), inputs=[api_key])
        custom_system_prompt.change(lambda x: save_to_storage("system_prompt", x), inputs=[custom_system_prompt])

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()
