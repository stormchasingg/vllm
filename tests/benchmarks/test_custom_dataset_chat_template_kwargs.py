# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path

import pytest

from vllm.benchmarks.datasets import CustomDataset, OpenAIMessagesDataset, get_samples


class _RecordingTokenizer:
    """Minimal tokenizer stub that records the kwargs forwarded to
    apply_chat_template, so we can assert chat_template_kwargs propagation
    without loading a real model/template."""

    def __init__(self) -> None:
        self.captured_kwargs: dict | None = None
        self.chat_template = "dummy-template"

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **kwargs,
    ) -> str:
        self.captured_kwargs = kwargs
        return conversation[0]["content"]

    def __call__(self, text: str):
        return argparse.Namespace(input_ids=list(range(len(text.split()))))


def _args(dataset_path: str, chat_template_kwargs) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name="custom",
        dataset_path=dataset_path,
        disable_shuffle=True,
        num_prompts=1,
        custom_output_len=32,
        skip_chat_template=False,
        chat_template_kwargs=chat_template_kwargs,
        no_oversample=False,
        seed=0,
        request_id_prefix="",
    )


def _write_one(path: Path) -> None:
    path.write_text(json.dumps({"prompt": "hello world"}) + "\n")


@pytest.mark.benchmark
def test_chat_template_kwargs_forwarded(tmp_path: Path) -> None:
    """--chat-template-kwargs must reach the client-side apply_chat_template."""
    jsonl = tmp_path / "data.jsonl"
    _write_one(jsonl)

    tok = _RecordingTokenizer()
    get_samples(_args(str(jsonl), {"thinking": True}), tok)

    assert tok.captured_kwargs == {"thinking": True}


@pytest.mark.benchmark
def test_chat_template_kwargs_default_is_noop(tmp_path: Path) -> None:
    """When not provided, no extra kwargs are passed (existing behavior)."""
    jsonl = tmp_path / "data.jsonl"
    _write_one(jsonl)

    tok = _RecordingTokenizer()
    get_samples(_args(str(jsonl), None), tok)

    assert tok.captured_kwargs == {}


@pytest.mark.benchmark
def test_extra_body_chat_template_kwargs_override_cli_kwargs(tmp_path: Path) -> None:
    jsonl = tmp_path / "data.jsonl"
    _write_one(jsonl)
    args = _args(str(jsonl), {"thinking": True})
    args.extra_body = {"chat_template_kwargs": {"thinking": False}}

    tok = _RecordingTokenizer()
    get_samples(args, tok)

    assert tok.captured_kwargs == {"thinking": False}


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "length_field", ["output_tokens", "max_tokens", "max_completion_tokens"]
)
def test_custom_messages_dataset_preserves_multimodal_requests(
    tmp_path: Path, length_field: str
) -> None:
    messages = [
        {"role": "system", "content": "hello world"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                },
            ],
        },
    ]
    jsonl = tmp_path / "messages.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "messages": messages,
                length_field: 17,
                "model": "source-model",
                "temperature": 0,
                "chat_template_kwargs": {"thinking": False},
            }
        )
        + "\n"
    )
    args = _args(str(jsonl), None)
    args.custom_output_len = -1
    args.request_id_prefix = "mm-"

    dataset = CustomDataset(dataset_path=str(jsonl), disable_shuffle=True)
    assert isinstance(dataset, OpenAIMessagesDataset)
    sample = get_samples(args, _RecordingTokenizer())[0]

    assert sample.chat_messages == messages
    assert sample.expected_output_len == 17
    assert sample.request_id == "mm-0"
    assert sample.request_overrides == {
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }


@pytest.mark.benchmark
def test_custom_messages_dataset_supports_skipping_tokenizer(tmp_path: Path) -> None:
    messages = [{"role": "user", "content": "hello world"}]
    jsonl = tmp_path / "messages.jsonl"
    jsonl.write_text(json.dumps({"messages": messages}) + "\n")

    sample = get_samples(_args(str(jsonl), None), None)[0]

    assert sample.chat_messages == messages
    assert sample.prompt == ""
    assert sample.prompt_len == 1


@pytest.mark.benchmark
@pytest.mark.parametrize("messages", [[], None, "not-a-list"])
def test_custom_messages_dataset_rejects_invalid_messages(
    tmp_path: Path, messages
) -> None:
    jsonl = tmp_path / "messages.jsonl"
    jsonl.write_text(json.dumps({"messages": messages}) + "\n")

    with pytest.raises(ValueError, match="must be a non-empty list"):
        get_samples(_args(str(jsonl), None), _RecordingTokenizer())
