"""Generate a README demo GIF illustrating streaming, tools, and citations."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "demo.gif"

W, H = 960, 540
BG = (250, 250, 252)
SIDEBAR = (245, 247, 250)
PANEL = (255, 255, 255)
BORDER = (220, 224, 230)
TEXT = (30, 35, 45)
MUTED = (100, 110, 125)
ACCENT = (59, 130, 246)
USER_BG = (239, 246, 255)
ASSIST_BG = (248, 250, 252)
GREEN = (34, 197, 94)
PURPLE = (139, 92, 246)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    radius: int = 12,
    outline: tuple[int, int, int] | None = None,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=1)


def _draw_sidebar(draw: ImageDraw.ImageDraw, highlight_upload: bool = False) -> None:
    draw.rectangle((0, 0, 220, H), fill=SIDEBAR)
    draw.line((220, 0, 220, H), fill=BORDER, width=1)
    title = _font(16, bold=True)
    body = _font(12)
    draw.text((16, 18), "Sessions", fill=TEXT, font=title)
    _rounded_rect(draw, (16, 52, 204, 88), fill=PANEL, radius=8, outline=BORDER)
    draw.text((28, 64), "Research notes chat", fill=TEXT, font=body)

    for idx, label in enumerate(("New", "Clear", "Delete")):
        x = 16 + idx * 62
        _rounded_rect(draw, (x, 100, x + 56, 128), fill=PANEL, radius=6, outline=BORDER)
        draw.text((x + 10, 108), label, fill=MUTED, font=body)

    draw.text((16, 150), "Knowledge Base", fill=TEXT, font=title)
    upload_fill = (219, 234, 254) if highlight_upload else PANEL
    _rounded_rect(draw, (16, 178, 204, 230), fill=upload_fill, radius=8, outline=ACCENT if highlight_upload else BORDER)
    draw.text((36, 192), "📄 report.pdf", fill=TEXT, font=body)
    draw.text((36, 212), "Upload & index", fill=MUTED, font=body)

    btn_fill = ACCENT if highlight_upload else (180, 190, 205)
    _rounded_rect(draw, (16, 242, 204, 272), fill=btn_fill, radius=8)
    draw.text((48, 252), "Index documents", fill=(255, 255, 255), font=body)


def _draw_header(draw: ImageDraw.ImageDraw) -> None:
    title = _font(22, bold=True)
    caption = _font(12)
    draw.text((240, 20), "Memory Agent Chat", fill=TEXT, font=title)
    draw.text(
        (240, 52),
        "Streaming · tool transparency · source citations",
        fill=MUTED,
        font=caption,
    )


def _draw_user_bubble(draw: ImageDraw.ImageDraw, text: str, y: int) -> None:
    body = _font(13)
    _rounded_rect(draw, (500, y, 920, y + 44), fill=USER_BG, radius=12, outline=BORDER)
    draw.text((520, y + 12), text, fill=TEXT, font=body)


def _draw_assistant_bubble(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    cursor: bool = False,
    tools: bool = False,
    sources: bool = False,
) -> int:
    body = _font(13)
    small = _font(11)
    display = text + ("▌" if cursor else "")
    lines = []
    current = ""
    for word in display.split():
        trial = f"{current} {word}".strip()
        if len(trial) > 58:
            if current:
                lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)

    bubble_h = 24 + len(lines) * 20
    _rounded_rect(draw, (240, y, 880, y + bubble_h), fill=ASSIST_BG, radius=12, outline=BORDER)
    for idx, line in enumerate(lines):
        draw.text((258, y + 12 + idx * 20), line, fill=TEXT, font=body)

    next_y = y + bubble_h + 8
    if tools:
        _rounded_rect(draw, (240, next_y, 880, next_y + 34), fill=(240, 249, 255), radius=8, outline=ACCENT)
        draw.text((256, next_y + 9), "🔧 Agent tools (2)  ·  retrieve_domain  ·  recall_memory", fill=ACCENT, font=small)
        next_y += 42
    if sources:
        _rounded_rect(draw, (240, next_y, 880, next_y + 34), fill=(240, 253, 244), radius=8, outline=GREEN)
        draw.text((256, next_y + 9), "📚 Sources (3)  ·  report.pdf  ·  chunk 0, 1, 2", fill=(22, 101, 52), font=small)
        next_y += 42
    return next_y


def _frame(
    user_text: str = "",
    assistant_text: str = "",
    cursor: bool = False,
    tools: bool = False,
    sources: bool = False,
    highlight_upload: bool = False,
    status: str = "",
) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    _draw_sidebar(draw, highlight_upload=highlight_upload)
    _draw_header(draw)

    y = 90
    if user_text:
        _draw_user_bubble(draw, user_text, y)
        y += 58

    if status:
        draw.text((240, y), status, fill=MUTED, font=_font(11))
        y += 24

    if assistant_text or cursor:
        _draw_assistant_bubble(
            draw,
            assistant_text,
            y,
            cursor=cursor,
            tools=tools,
            sources=sources,
        )

    return img


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    frames = [
        (_frame(highlight_upload=True), 900),
        (_frame(highlight_upload=True, status="Indexing report.pdf…"), 700),
        (_frame(user_text="Summarize the key points from report.pdf"), 900),
        (_frame(user_text="Summarize the key points from report.pdf", status="Searching knowledge base…"), 500),
        (_frame(user_text="Summarize the key points from report.pdf", assistant_text="The document covers three main", cursor=True), 350),
        (_frame(user_text="Summarize the key points from report.pdf", assistant_text="The document covers three main themes: multimodal RAG, agent memory, and", cursor=True), 350),
        (_frame(
            user_text="Summarize the key points from report.pdf",
            assistant_text="The document covers three main themes: multimodal RAG, agent memory, and checkpointed sessions with grounded citations.",
            tools=True,
        ), 900),
        (_frame(
            user_text="Summarize the key points from report.pdf",
            assistant_text="The document covers three main themes: multimodal RAG, agent memory, and checkpointed sessions with grounded citations.",
            tools=True,
            sources=True,
        ), 1400),
        (_frame(
            user_text="Summarize the key points from report.pdf",
            assistant_text="The document covers three main themes: multimodal RAG, agent memory, and checkpointed sessions with grounded citations.",
            tools=True,
            sources=True,
        ), 900),
    ]

    images = [frame for frame, _ in frames]
    durations = [duration for _, duration in frames]

    images[0].save(
        OUTPUT,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
