"""Tests for SSE parsing and Responses API reconstruction."""

import json

import pytest

from codex_probe.sse import (
    ChatCompletionsReassembler,
    ResponsesReassembler,
    SseEvent,
    SseParser,
)


def _as_sse(payload: dict[str, object]) -> bytes:
    """Encode one dictionary as an SSE data event."""

    data = json.dumps(payload)
    return f"data: {data}\n\n".encode("utf-8")


def test_incomplete_event_waits_for_more_bytes() -> None:
    parser = SseParser()

    first_result = parser.feed(
        b'data: {"type":"response.'
    )
    second_result = parser.feed(
        b'completed"}\n\n'
    )

    assert first_result == []
    assert second_result == [
        SseEvent(
            data='{"type":"response.completed"}',
        )
    ]


def test_multiple_events_can_arrive_in_one_chunk() -> None:
    parser = SseParser()

    events = parser.feed(
        b'data: {"number":1}\n\n'
        b'data: {"number":2}\n\n'
    )

    assert events == [
        SseEvent(data='{"number":1}'),
        SseEvent(data='{"number":2}'),
    ]


def test_event_field_is_preserved() -> None:
    parser = SseParser()

    events = parser.feed(
        b"event: response.completed\n"
        b'data: {"id":"response-123"}\n\n'
    )

    assert events == [
        SseEvent(
            event="response.completed",
            data='{"id":"response-123"}',
        )
    ]


@pytest.mark.parametrize(
    "raw_event",
    [
        b"data: hello\n\n",
        b"data: hello\r\n\r\n",
        b"data: hello\r\r",
    ],
)
def test_supported_line_endings_are_parsed(
    raw_event: bytes,
) -> None:
    parser = SseParser()

    assert parser.feed(raw_event) == [
        SseEvent(data="hello")
    ]


def test_boundary_can_be_split_between_chunks() -> None:
    parser = SseParser()

    first_result = parser.feed(
        b"data: hello\r\n\r"
    )
    second_result = parser.feed(b"\n")

    assert first_result == []
    assert second_result == [
        SseEvent(data="hello")
    ]


def test_utf8_character_can_be_split_between_chunks() -> None:
    parser = SseParser()

    complete_event = (
        'data: {"text":"नमस्ते"}\n\n'
    ).encode("utf-8")

    character = "न".encode("utf-8")
    character_position = complete_event.index(character)
    split_position = character_position + 1

    first_result = parser.feed(
        complete_event[:split_position]
    )
    second_result = parser.feed(
        complete_event[split_position:]
    )

    assert first_result == []
    assert second_result == [
        SseEvent(data='{"text":"नमस्ते"}')
    ]


def test_comments_and_unsupported_fields_are_ignored() -> None:
    parser = SseParser()

    events = parser.feed(
        b": keep-alive\n"
        b"id: event-123\n"
        b"retry: 1000\n"
        b"data: payload\n\n"
    )

    assert events == [
        SseEvent(data="payload")
    ]


def test_multiple_data_lines_are_joined() -> None:
    parser = SseParser()

    events = parser.feed(
        b"data: first line\n"
        b"data: second line\n\n"
    )

    assert events == [
        SseEvent(
            data="first line\nsecond line",
        )
    ]


def test_finish_returns_unterminated_final_event() -> None:
    parser = SseParser()

    assert parser.feed(
        b"data: final payload"
    ) == []

    assert parser.finish() == [
        SseEvent(data="final payload")
    ]


def test_comment_only_event_is_not_returned() -> None:
    parser = SseParser()

    events = parser.feed(
        b": keep-alive\n\n"
    )

    assert events == []


def test_feed_after_finish_is_rejected() -> None:
    parser = SseParser()
    parser.finish()

    with pytest.raises(
        RuntimeError,
        match="cannot feed a finished SSE parser",
    ):
        parser.feed(b"data: too late\n\n")


def test_second_finish_is_rejected() -> None:
    parser = SseParser()
    parser.finish()

    with pytest.raises(
        RuntimeError,
        match="SSE parser is already finished",
    ):
        parser.finish()


def test_responses_stream_is_reassembled() -> None:
    parser = SseParser()
    reassembler = ResponsesReassembler()

    expected_response = {
        "id": "resp-123",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Task completed",
                    }
                ],
            }
        ],
    }

    stream = b"".join(
        [
            _as_sse(
                {
                    "type": "response.created",
                    "response": {
                        "id": "resp-123",
                        "status": "in_progress",
                    },
                }
            ),
            _as_sse(
                {
                    "type": "response.output_text.delta",
                    "delta": "Task completed",
                }
            ),
            _as_sse(
                {
                    "type": "response.completed",
                    "response": expected_response,
                }
            ),
        ]
    )

    chunks = [
        stream[:17],
        stream[17:63],
        stream[63:],
    ]

    for chunk in chunks:
        for event in parser.feed(chunk):
            reassembler.accept(event)

    for event in parser.finish():
        reassembler.accept(event)

    assert reassembler.finish() == expected_response


@pytest.mark.parametrize(
    "event_type",
    [
        "response.failed",
        "response.incomplete",
    ],
)
def test_failed_and_incomplete_responses_are_captured(
    event_type: str,
) -> None:
    reassembler = ResponsesReassembler()
    response = {
        "id": "resp-123",
        "status": event_type.removeprefix("response."),
    }

    reassembler.accept(
        SseEvent(
            data=json.dumps(
                {
                    "type": event_type,
                    "response": response,
                }
            )
        )
    )

    assert reassembler.finish() == response


def test_done_marker_is_ignored() -> None:
    reassembler = ResponsesReassembler()

    reassembler.accept(SseEvent(data="[DONE]"))

    with pytest.raises(
        ValueError,
        match="stream ended without a terminal response",
    ):
        reassembler.finish()


def test_terminal_event_requires_response_object() -> None:
    reassembler = ResponsesReassembler()

    with pytest.raises(
        ValueError,
        match="terminal event must contain a response object",
    ):
        reassembler.accept(
            SseEvent(
                data=json.dumps(
                    {"type": "response.completed"}
                )
            )
        )


def test_responses_event_must_be_json_object() -> None:
    reassembler = ResponsesReassembler()

    with pytest.raises(
        ValueError,
        match="must be a JSON object",
    ):
        reassembler.accept(
            SseEvent(data='["not", "an", "object"]')
        )


def test_missing_terminal_response_is_rejected() -> None:
    reassembler = ResponsesReassembler()

    reassembler.accept(
        SseEvent(
            data=json.dumps(
                {"type": "response.output_text.delta"}
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="stream ended without a terminal response",
    ):
        reassembler.finish()

def test_chat_text_stream_is_reassembled() -> None:
    reassembler = ChatCompletionsReassembler()

    reassembler.accept(
        SseEvent(
            data=json.dumps(
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "qwen2.5-coder",
                    "system_fingerprint": "fingerprint-123",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )
    )

    reassembler.accept(
        SseEvent(
            data=json.dumps(
                {
                    "id": "chatcmpl-123",
                    "created": 1234567890,
                    "model": "qwen2.5-coder",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": "Hello",
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )
    )

    reassembler.accept(
        SseEvent(
            data=json.dumps(
                {
                    "id": "chatcmpl-123",
                    "created": 1234567890,
                    "model": "qwen2.5-coder",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": " world",
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )
    )

    reassembler.accept(
        SseEvent(
            data=json.dumps(
                {
                    "id": "chatcmpl-123",
                    "created": 1234567890,
                    "model": "qwen2.5-coder",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        )
    )

    reassembler.accept(
        SseEvent(
            data=json.dumps(
                {
                    "id": "chatcmpl-123",
                    "created": 1234567890,
                    "model": "qwen2.5-coder",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                }
            )
        )
    )

    reassembler.accept(SseEvent(data="[DONE]"))

    assert reassembler.finish() == {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen2.5-coder",
        "system_fingerprint": "fingerprint-123",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello world",
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
    }


def test_chat_tool_call_fragments_are_reassembled() -> None:
    reassembler = ChatCompletionsReassembler()

    reassembler.accept(
        SseEvent(
            data=json.dumps(
                {
                    "id": "chatcmpl-tool",
                    "created": 1234567890,
                    "model": "qwen2.5-coder",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-123",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )
    )

    reassembler.accept(
        SseEvent(
            data=json.dumps(
                {
                    "id": "chatcmpl-tool",
                    "created": 1234567890,
                    "model": "qwen2.5-coder",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": '"config.py"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )
    )

    reassembler.accept(
        SseEvent(
            data=json.dumps(
                {
                    "id": "chatcmpl-tool",
                    "created": 1234567890,
                    "model": "qwen2.5-coder",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            )
        )
    )

    assert reassembler.finish() == {
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen2.5-coder",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-123",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"config.py"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
                "logprobs": None,
            }
        ],
    }


def test_chat_choices_are_ordered_by_index() -> None:
    reassembler = ChatCompletionsReassembler()

    reassembler.accept(
        SseEvent(
            data=json.dumps(
                {
                    "id": "chatcmpl-multiple",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 1,
                            "delta": {
                                "role": "assistant",
                                "content": "Second",
                            },
                            "finish_reason": "stop",
                        },
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "First",
                            },
                            "finish_reason": "stop",
                        },
                    ],
                }
            )
        )
    )

    response = reassembler.finish()
    choices = response["choices"]

    assert isinstance(choices, list)
    assert choices[0]["index"] == 0
    assert choices[1]["index"] == 1


@pytest.mark.parametrize(
    "choices",
    [
        "not-a-list",
        {"index": 0},
        42,
    ],
)
def test_chat_choices_must_be_a_list(
    choices: object,
) -> None:
    reassembler = ChatCompletionsReassembler()

    with pytest.raises(
        ValueError,
        match="Chat stream choices must be a list",
    ):
        reassembler.accept(
            SseEvent(
                data=json.dumps(
                    {
                        "choices": choices,
                    }
                )
            )
        )


def test_chat_choice_must_be_an_object() -> None:
    reassembler = ChatCompletionsReassembler()

    with pytest.raises(
        ValueError,
        match="Chat stream choice must be an object",
    ):
        reassembler.accept(
            SseEvent(
                data=json.dumps(
                    {
                        "choices": [
                            "not-an-object",
                        ],
                    }
                )
            )
        )


def test_chat_choice_requires_integer_index() -> None:
    reassembler = ChatCompletionsReassembler()

    with pytest.raises(
        ValueError,
        match="must have an integer index",
    ):
        reassembler.accept(
            SseEvent(
                data=json.dumps(
                    {
                        "choices": [
                            {
                                "index": "zero",
                                "delta": {
                                    "content": "Hello",
                                },
                            }
                        ],
                    }
                )
            )
        )


def test_chat_tool_calls_must_be_a_list() -> None:
    reassembler = ChatCompletionsReassembler()

    with pytest.raises(
        ValueError,
        match="Chat stream tool_calls must be a list",
    ):
        reassembler.accept(
            SseEvent(
                data=json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": "not-a-list",
                                },
                            }
                        ],
                    }
                )
            )
        )


def test_chat_tool_call_requires_integer_index() -> None:
    reassembler = ChatCompletionsReassembler()

    with pytest.raises(
        ValueError,
        match="Chat tool call must have an integer index",
    ):
        reassembler.accept(
            SseEvent(
                data=json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": "zero",
                                            "function": {
                                                "name": "read_file",
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                )
            )
        )


def test_chat_stream_without_choices_is_rejected() -> None:
    reassembler = ChatCompletionsReassembler()

    reassembler.accept(SseEvent(data="[DONE]"))

    with pytest.raises(
        ValueError,
        match="Chat stream ended without any choices",
    ):
        reassembler.finish()