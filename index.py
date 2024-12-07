import os
import shutil
import glob
import cv2
import numpy as np
import math

from disp_util import display  # 結果表示用モジュール


# 画像の読み込み(横幅640を基準にサイズ補正される)
def read_image(filepath):
    basename = os.path.basename(filepath).split(".")[0]
    image = cv2.imread(filepath)
    org_h, org_w, _ = image.shape
    fx = 640 / org_w
    image = cv2.resize(image, None, fx=fx, fy=fx)
    new_h, new_w, _ = image.shape
    print(f"read_image  {org_w} x {org_h} => fx: {fx} => {new_w} x {new_h}")
    return basename, image


# ２値化画像の作成
def create_binarize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 220, cv2.THRESH_BINARY)
    return binary


# 気圧計の外円を検出
def detect_circle(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        if 110 < radius < 130:
            print(f"detect_circle center:{center} radius:{radius}")
            return center, radius
    return (0, 0), 0


# エッジ画像を生成
def create_edges_image(binary_image):
    return cv2.Canny(binary_image, 50, 150, apertureSize=3)


# 直線検出
def detect_lines(edges_image):
    lines = cv2.HoughLinesP(
        edges_image, 1, np.pi / 180, threshold=70, minLineLength=50, maxLineGap=10
    )
    print(f"detect_lines lines: {lines}")
    return lines


# 中央部を表現する矩形
def get_center_area(center, margin):
    return (center[0] - margin, center[1] - margin, margin * 2, margin * 2)


# メータの針を検出
def detect_pointer(center_area, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if crossing_detection((x1, y1, x2, y2), center_area):
            print(f"detect_pointer ({x1},{y1}) ({x2},{y2})")
            return (x1, y1), (x2, y2)
    return None


# 針先の座標を取得
def get_pointer_coordinates(pointer, center):
    c = np.array(center)
    a = np.array(pointer[0])
    b = np.array(pointer[1])
    return pointer[0] if np.linalg.norm(c - a) > np.linalg.norm(c - b) else pointer[1]


# 針先の角度を取得
def get_deg(point, center):
    x = point[0] - center[0]
    y = (point[1] - center[1]) * -1
    rad = np.arctan2(x, y)
    deg = math.degrees(rad) + 180
    return round(deg, 2)


# 針先の角度をメータ値に変換する
def convert_to_meter_value(deg, start_deg, end_deg):
    value = (deg - start_deg) / (end_deg - start_deg)
    return round(value, 2)


# 直線の交差検���
def crossing_detection_line(p1, p2, q1, q2):
    def is_clockwise(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return is_clockwise(p1, q1, q2) != is_clockwise(p2, q1, q2) and is_clockwise(
        p1, p2, q1
    ) != is_clockwise(p1, p2, q2)


# 直線と矩形の交差検出
def crossing_detection(line, rect):
    x1, y1, x2, y2 = line
    rx, ry, rw, rh = rect
    rect_lines = [
        ((rx, ry), (rx + rw, ry)),
        ((rx, ry), (rx, ry + rh)),
        ((rx + rw, ry), (rx + rw, ry + rh)),
        ((rx, ry + rh), (rx + rw, ry + rh)),
    ]
    for rect_line in rect_lines:
        if crossing_detection_line((x1, y1), (x2, y2), rect_line[0], rect_line[1]):
            return True
    return False


def process_image(filepath, output_path):
    start_deg = 45
    end_deg = 315
    center_margin = 15

    basename, image = read_image(filepath)
    binary_image = create_binarize_image(image)
    cv2.imwrite(f"{output_path}/{basename}_binary.png", binary_image)

    center, radius = detect_circle(binary_image)
    edges_image = create_edges_image(binary_image)
    cv2.imwrite(f"{output_path}/{basename}_edge.png", edges_image)

    lines = detect_lines(edges_image)
    center_area = get_center_area(center, center_margin)
    pointer = detect_pointer(center_area, lines)
    point = get_pointer_coordinates(pointer, center)
    deg = get_deg(point, center)
    value = convert_to_meter_value(deg, start_deg, end_deg)

    display(image, center, radius, center_area, pointer, deg, value, start_deg, end_deg)
    cv2.imwrite(f"{output_path}/{basename}_result.png", image)

    cv2.imshow("Image", image)
    cv2.waitKey(0)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = f"{current_dir}/input"
    output_path = f"{current_dir}/output"

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    image_files = glob.glob(f"{input_path}/*.png")
    for filepath in image_files:
        print(filepath)
        try:
            process_image(filepath, output_path)
        except Exception as e:
            print("ERROR:", e.args)

    cv2.destroyAllWindows()


main()
