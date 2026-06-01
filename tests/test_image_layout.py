
from io import BytesIO

from PIL import Image, ImageDraw

from pcb_sustainability.image_layout import parse_pcb_image, score_pcb


def _make_demo_image() -> BytesIO:
    img = Image.new("RGB", (640, 360), (38, 112, 64))
    draw = ImageDraw.Draw(img)

    # Board border and a few component-like shapes.
    draw.rounded_rectangle((24, 24, 616, 336), radius=18, outline=(235, 235, 235), width=8)
    draw.rectangle((80, 80, 170, 130), fill=(20, 20, 20))
    draw.rectangle((210, 70, 320, 155), fill=(12, 12, 12))
    draw.rectangle((360, 90, 500, 140), fill=(18, 18, 18))
    draw.ellipse((110, 190, 190, 270), fill=(230, 230, 230))
    draw.rectangle((250, 205, 420, 250), fill=(30, 30, 30))
    draw.rectangle((470, 210, 560, 275), fill=(240, 240, 240))
    draw.text((90, 40), "U1", fill=(255, 255, 255))
    draw.text((250, 40), "J1", fill=(255, 255, 255))
    draw.text((470, 165), "C3", fill=(255, 255, 255))

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    buf.name = "demo_pcb.png"
    return buf


def test_parse_pcb_image_extracts_reasonable_features():
    features = parse_pcb_image(_make_demo_image())
    assert features.board_area_mm2 and features.board_area_mm2 > 0
    assert features.component_count >= 1
    assert features.layer_count >= 2
    assert features.warnings


def test_score_pcb_accepts_image_features():
    features = parse_pcb_image(_make_demo_image())
    score = score_pcb(features)
    assert 0 <= score.score <= 100
    assert 0 <= score.accessibility <= 100
    assert score.recommendations
