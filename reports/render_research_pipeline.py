from pathlib import Path
from math import atan2, cos, sin, pi

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("research_pipeline_two_column_fixed.png")
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

SCALE = 2
W, H = 780 * SCALE, 520 * SCALE
NODE_W, NODE_H = 300 * SCALE, 58 * SCALE
LEFT_X, RIGHT_X = 40 * SCALE, 440 * SCALE
YS = [30 * SCALE, 125 * SCALE, 220 * SCALE, 315 * SCALE, 410 * SCALE]

BG = "white"
NODE_FILL = "#f8fafc"
STROKE = "#334155"
TEXT = "#111827"


def arrow(draw, points, width=4 * SCALE, color=STROKE):
    draw.line(points, fill=color, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = atan2(y2 - y1, x2 - x1)
    size = 8 * SCALE
    left = (x2 - size * cos(angle - pi / 6), y2 - size * sin(angle - pi / 6))
    right = (x2 - size * cos(angle + pi / 6), y2 - size * sin(angle + pi / 6))
    draw.polygon([(x2, y2), left, right], fill=color)


def centered_text(draw, box, text, font):
    x, y, w, h = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x + (w - tw) / 2, y + (h - th) / 2 - 2 * SCALE), text, font=font, fill=TEXT)


def node(draw, x, y, label, font):
    draw.rounded_rectangle(
        (x, y, x + NODE_W, y + NODE_H),
        radius=8 * SCALE,
        fill=NODE_FILL,
        outline=STROKE,
        width=2 * SCALE,
    )
    centered_text(draw, (x, y, NODE_W, NODE_H), label, font)


def main():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 17 * SCALE)

    left_labels = [
        "원천 데이터 수집",
        "전처리 및 통합 데이터셋 생성",
        "장기 상관관계 분석",
        "Anomaly Block 탐지",
        "SHAP 기반 요인 분석",
    ]
    right_labels = [
        "M2 구성요소 분석",
        "단기 유동성 임계점 검증",
        "Granger Causality 검정",
        "LSTM/Hybrid 예측 검증",
        "환율 영향 확장 분석",
    ]

    for y, label in zip(YS, left_labels):
        node(draw, LEFT_X, y, label, font)
    for y, label in zip(YS, right_labels):
        node(draw, RIGHT_X, y, label, font)

    left_mid_x = LEFT_X + NODE_W / 2
    right_mid_x = RIGHT_X + NODE_W / 2
    for y1, y2 in zip(YS[:-1], YS[1:]):
        arrow(draw, [(left_mid_x, y1 + NODE_H), (left_mid_x, y2 - 4 * SCALE)])
        arrow(draw, [(right_mid_x, y1 + NODE_H), (right_mid_x, y2 - 4 * SCALE)])

    gap_x = 390 * SCALE
    arrow(
        draw,
        [
            (LEFT_X + NODE_W, YS[-1] + NODE_H / 2),
            (gap_x, YS[-1] + NODE_H / 2),
            (gap_x, YS[0] + NODE_H / 2),
            (RIGHT_X - 4 * SCALE, YS[0] + NODE_H / 2),
        ],
    )

    img.save(OUT)


if __name__ == "__main__":
    main()
