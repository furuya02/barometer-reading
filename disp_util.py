import cv2
import numpy as np

BASE_LINE_COLOR = (0, 0, 255)  # 基準線の色
POINTER_COLOR = (255, 255, 0)  # メータ針の色
CENTER_AREA_COLOR = (255, 255, 255)  # 中央部の矩形の色
TEXT_COLOR = (255, 255, 255)  # テキストの色
VALUE_COLOR = (255, 255, 0)  # メータ値の色


def __draw_base_line(image, center, radius, angle, color):
    # angle　は　+90　することで、真下を0度となるように変換して利用する
    rcosθ = radius * np.cos(np.radians(angle + 90))
    rsinθ = radius * np.sin(np.radians(angle + 90))
    t = np.array([center[0] + rcosθ, center[1] + rsinθ])
    cv2.line(image, center, (int(t[0]), int(t[1])), color, 2)
    cv2.putText(
        image,
        f"{angle}",
        (int(t[0]), int(t[1]) + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
    )


def __draw_text(image, text, position, color, font_scale=0.7):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
    )


# 結果の描画
def display(
    image,
    center,
    radius,
    center_area,
    pointer,
    deg,
    value,
    start_deg,
    end_deg,
):

    # メータ針を描画
    cv2.line(image, pointer[0], pointer[1], POINTER_COLOR, 2)

    # 基準線の表示
    __draw_base_line(image, center, radius, start_deg, BASE_LINE_COLOR)
    __draw_base_line(image, center, radius, end_deg, BASE_LINE_COLOR)

    # 中央部を表現する矩形を描画
    cv2.rectangle(
        image,
        (center_area[0], center_area[1]),
        (center_area[0] + center_area[2], center_area[1] + center_area[3]),
        CENTER_AREA_COLOR,
        2,
    )
    # 気圧計の外円を描画
    cv2.circle(image, center, radius, (0, 255, 0), 2)

    # 結果を表示
    y = 30
    __draw_text(
        image,
        f"center: {center}",
        (10, y),
        TEXT_COLOR,
    )
    __draw_text(image, f"pointer: {pointer}", (10, y + 30), TEXT_COLOR)
    __draw_text(image, f"deg: {deg}", (10, y + 60), TEXT_COLOR)
    __draw_text(image, f"value: {value}", (10, y + 90), VALUE_COLOR)
