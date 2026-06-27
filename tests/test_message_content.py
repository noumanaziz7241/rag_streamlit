from memory_agent.utils.message_content import extract_text_content


def test_extract_plain_string():
    assert extract_text_content("Hello") == "Hello"


def test_extract_gemini_text_blocks():
    content = [
        {
            "type": "text",
            "text": "The WiseHealth scoring mechanism",
            "extras": {"signature": "CnMBDDnWx0Gv4wUmYhuEiyjf00XjqzG0U1tLcu+3M2a"},
        },
        " across several specific health domains.",
    ]
    assert extract_text_content(content) == (
        "The WiseHealth scoring mechanism across several specific health domains."
    )


def test_extract_skips_non_text_blocks():
    content = [
        {"type": "thinking", "text": "internal reasoning"},
        {"type": "text", "text": "Visible answer."},
    ]
    assert extract_text_content(content) == "Visible answer."


def test_extract_none_and_empty():
    assert extract_text_content(None) == ""
    assert extract_text_content([]) == ""


def test_recover_stringified_gemini_blocks():
    raw = str([
        {
            "type": "text",
            "text": "Hello",
            "extras": {"signature": "abc123"},
        },
        " world.",
    ])
    assert extract_text_content(raw) == "Hello world."
