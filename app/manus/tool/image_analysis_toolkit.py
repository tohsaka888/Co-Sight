# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

from dotenv import load_dotenv

from ..gate.format_gate import format_check

load_dotenv()
import os
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
import asyncio
from .file_download_toolkit import download_file, get_filename_from_url


class VisionTool():
    def __init__(self, llm_config):
        self.llm_config = llm_config

    name: str = "Vision Tool"
    description: str = (
        "This tool uses OpenAI's Vision API to describe the contents of an Image."
    )
    _client: OpenAI = None

    @property
    def client(self) -> OpenAI:
        llm_config = {"api_key": self.llm_config['api_key'],
                      "base_url": self.llm_config['base_url']
                      }
        """Cached ChatOpenAI client instance."""
        if self._client is None:
            self._client = OpenAI(**llm_config)
        return self._client

    #  Base64 编码格式
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def parse_param(self, image_path_url):
        img_url = ''
        if image_path_url.startswith('http://') or image_path_url.startswith('https://'):
            workspace_path = os.getenv("WORKSPACE_PATH") or os.getcwd()
            file_name = get_filename_from_url(image_path_url)
            absolute_path = os.path.join(workspace_path, file_name)
            download_file(image_path_url, absolute_path)
            base64_image = self.encode_image(absolute_path)
            img_url = f"data:image/png;base64,{base64_image}"
        else:
            base64_image = self.encode_image(image_path_url)
            img_url = f"data:image/png;base64,{base64_image}"
        return img_url

    async def _run(self, image_path_url, task_prompt):
        img_url = self.parse_param(image_path_url)
        completion = self.client.chat.completions.create(
            extra_headers={'Content-Type': 'application/json',
                           'Authorization': 'Bearer %s' % self.llm_config['api_key']},
            model=self.llm_config['model'],
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text",
                         "text": """
                         You are an exceptionally helpful and honest AI assistant. Your primary directive is to provide detailed, comprehensive, and accurate answers to user queries. You must never omit critical information or key details. Under no circumstances is it allowed to fabricate the result 
                         """
                            },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_url},
                        },
                        {"type": "text", "text": task_prompt},
                    ],
                },
            ],
            temperature=self.llm_config['temperature'],
            # 设置输出数据的模态，当前支持两种：["text","audio"]、["text"]
            modalities=["text"],
            # audio={"voice": "Cherry", "format": "wav"},
            # stream 必须设置为 True，否则会报错
            stream=True,
            stream_options={"include_usage": True},
        )
        full_response = ""
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "audio") and delta.audio:
                    try:
                        if delta.audio['transcript']:
                            full_response += delta.audio['transcript']
                    except Exception as ex:
                        pass
                if hasattr(delta, "content") and delta.content:
                    try:
                        full_response += delta.content
                    except Exception as ex:
                        pass
            else:
                pass

        print(f'ask_question_about_image result {full_response}')
        return full_response

    @format_check()
    def ask_question_about_image(self, image_path_url, task_prompt):
        print(f"Using Tool: {self.name}")
        return asyncio.run(self._run(image_path_url, task_prompt))
