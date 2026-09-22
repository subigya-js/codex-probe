"""Tests for incremental server-sent event parsing."""

import pytest

from codex_probe.sse import SseEvent, SseParser


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
