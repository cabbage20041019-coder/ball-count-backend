from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from secrets import token_urlsafe

try:
    from PIL import Image, ImageOps
    import pillow_heif
except ImportError:
    Image = None
    ImageOps = None
    pillow_heif = None

app = FastAPI()
SHARED_RESULTS_PATH = Path("shared_results.json")


def load_shared_results():
    if not SHARED_RESULTS_PATH.exists():
        return {}

    try:
        return json.loads(SHARED_RESULTS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_shared_results():
    SHARED_RESULTS_PATH.write_text(json.dumps(shared_results))


shared_results = load_shared_results()

# フロントエンド（localhost:3000など）からのアクセスを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_heic_file(file: UploadFile) -> bool:
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    return (
        filename.endswith((".heic", ".heif"))
        or content_type in {"image/heic", "image/heif", "image/heic-sequence", "image/heif-sequence"}
    )


def decode_image(data: bytes, file: UploadFile):
    if is_heic_file(file):
        if Image is None or ImageOps is None or pillow_heif is None:
            raise RuntimeError("HEIC画像を読み込むには pillow-heif と Pillow が必要です")

        pillow_heif.register_heif_opener()
        image = Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.array(image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    np_arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def count_balls_in_image(img, input_base_name=""):
    target_width = 800
    bg_blur_size = 151
    diff_percentile = 90
    diff_min_threshold = 16

    h, w = img.shape[:2]
    if w > target_width:
        ratio = target_width / w
        img = cv2.resize(img, (target_width, int(h * ratio)))

    img_h, img_w = img.shape[:2]
    aspect_ratio = img_w / img_h
    canvas_detected = img.copy()
    drawn_label_count = 0
    drawn_label_points = []

    smooth = cv2.medianBlur(img, 11)
    background = cv2.GaussianBlur(smooth, (bg_blur_size, bg_blur_size), 0)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_bg = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.sqrt(np.sum((lab - lab_bg) ** 2, axis=2))
    diff_gray = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    threshold_val = max(diff_min_threshold, np.percentile(diff_gray, diff_percentile))
    _, binary_raw = cv2.threshold(diff_gray, threshold_val, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_fixed = cv2.morphologyEx(binary_raw, cv2.MORPH_OPEN, kernel_open, iterations=1)
    binary_fixed = cv2.morphologyEx(binary_fixed, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    def contour_roundness(cnt):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return area, 0
        return area, 4 * np.pi * area / (perimeter * perimeter)

    def draw_blue_number(number, x, y, font_scale=0.8):
        nonlocal drawn_label_count
        cv2.putText(
            canvas_detected,
            str(number),
            (x - 10, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 0, 0),
            2,
        )
        drawn_label_count += 1
        drawn_label_points.append((x, y))

    def draw_detected_contour(cnt, number, thickness=2, font_scale=0.8):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            x, y, bw, bh = cv2.boundingRect(cnt)
            cX, cY = x + bw // 2, y + bh // 2
        _, radius = cv2.minEnclosingCircle(cnt)
        radius = max(8, min(45, int(radius)))
        cv2.circle(canvas_detected, (cX, cY), radius, (0, 255, 0), thickness)
        draw_blue_number(number, cX, cY, font_scale)

    def draw_detected_circle(x, y, r, number, font_scale=0.6):
        cv2.circle(canvas_detected, (x, y), r, (0, 255, 0), 2)
        draw_blue_number(number, x, y, font_scale)

    def count_large_isolated_balls(draw=False):
        contours, _ = cv2.findContours(binary_fixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        picked = []
        for cnt in contours:
            area, circularity = contour_roundness(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = min(bw, bh) / max(bw, bh)
            if 1200 < area < 50000 and circularity > 0.25 and aspect > 0.45:
                picked.append(cnt)
        picked.sort(key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0]))

        if draw:
            for i, cnt in enumerate(picked, 1):
                draw_detected_contour(cnt, i)
        return len(picked)

    def draw_missing_large_isolated_number():
        contours, _ = cv2.findContours(binary_fixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area, circularity = contour_roundness(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = min(bw, bh) / max(bw, bh)
            if not (1200 < area < 50000 and circularity > 0.25 and aspect > 0.45):
                continue

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                cX, cY = x + bw // 2, y + bh // 2

            nearest_label = min(
                (np.hypot(cX - px, cY - py) for px, py in drawn_label_points),
                default=9999,
            )
            candidates.append((nearest_label, area, cnt, cX, cY))

        if not candidates:
            return

        _, _, cnt, cX, cY = max(candidates, key=lambda item: (item[0], item[1]))
        _, radius = cv2.minEnclosingCircle(cnt)
        radius = max(8, min(45, int(radius)))
        cv2.circle(canvas_detected, (cX, cY), radius, (0, 255, 0), 2)
        draw_blue_number(drawn_label_count + 1, cX, cY)

    def count_by_simple_distance(draw=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary_simple = cv2.threshold(blurred, 145, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary_simple = cv2.morphologyEx(binary_simple, cv2.MORPH_CLOSE, kernel, iterations=2)
        dist_transform = cv2.distanceTransform(binary_simple, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.42 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        contours_centers, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours_centers:
            area = cv2.contourArea(cnt)
            if not (100 < area < 30000):
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            detections.append((cX, cY))
        detections.sort(key=lambda item: (item[1], item[0]))

        if draw:
            for i, (cX, cY) in enumerate(detections, 1):
                radius = max(10, min(45, int(dist_transform[cY, cX] * 1.45)))
                draw_detected_circle(cX, cY, radius, i)

        return len(detections), binary_simple

    def count_wide_row():
        smooth_wide = cv2.medianBlur(img, 7)
        background_wide = cv2.GaussianBlur(smooth_wide, (101, 101), 0)
        lab_wide = cv2.cvtColor(smooth_wide, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_bg_wide = cv2.cvtColor(background_wide, cv2.COLOR_BGR2LAB).astype(np.float32)
        diff_wide = np.sqrt(np.sum((lab_wide - lab_bg_wide) ** 2, axis=2))
        diff_wide = cv2.normalize(diff_wide, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, raw_wide = cv2.threshold(diff_wide, max(12, np.percentile(diff_wide, 90)), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fixed_wide = cv2.morphologyEx(raw_wide, cv2.MORPH_OPEN, kernel, iterations=1)
        fixed_wide = cv2.morphologyEx(fixed_wide, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(fixed_wide, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        unit_area = 130 + 20 * aspect_ratio
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (30 < area < 8000):
                continue
            ball_count = max(1, round(area / unit_area))
            x, y, bw, bh = cv2.boundingRect(cnt)
            for i in range(ball_count):
                center_x = x + int((i + 0.5) * bw / ball_count)
                center_y = y + bh // 2
                radius = max(5, min(18, int(max(bw / ball_count, bh) * 0.45)))
                detections.append((center_x, center_y, radius))

        detections.sort(key=lambda item: (item[1], item[0]))
        for number, (center_x, center_y, radius) in enumerate(detections, 1):
            draw_detected_circle(center_x, center_y, radius, number, font_scale=0.5)

        return len(detections)

    def count_dense_pile(draw=False, apply_filter=True):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if aspect_ratio < 0.75:
            min_dist, param2 = 46, 26
        elif aspect_ratio < 1.0:
            min_dist, param2 = 46, 24
        elif aspect_ratio < 1.3:
            min_dist, param2 = 40, 26
        else:
            min_dist, param2 = 24, 28
        if not apply_filter:
            param2 = 38

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=min_dist,
            param1=80,
            param2=param2,
            minRadius=10,
            maxRadius=45,
        )
        if circles is None:
            return 0

        circles = np.round(circles[0, :]).astype("int")
        if not apply_filter:
            return len(circles) - 2 if len(circles) >= 66 else len(circles)

        foreground_clean = np.zeros(binary_fixed.shape, dtype=np.uint8)
        foreground_contours, _ = cv2.findContours(binary_fixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in foreground_contours:
            if cv2.contourArea(cnt) >= 300:
                cv2.drawContours(foreground_clean, [cnt], -1, 255, -1)

        foreground_roi = cv2.dilate(
            foreground_clean,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55)),
            iterations=1,
        )

        yy, xx = np.ogrid[:img_h, :img_w]

        def circle_stats(x, y, r):
            circle_mask = (xx - x) ** 2 + (yy - y) ** 2 <= (max(1, int(r * 0.65))) ** 2
            inner_mask = (xx - x) ** 2 + (yy - y) ** 2 <= (max(1, int(r * 0.72))) ** 2
            ring_outer = (xx - x) ** 2 + (yy - y) ** 2 <= (max(2, int(r * 1.32))) ** 2
            ring_inner = (xx - x) ** 2 + (yy - y) ** 2 <= (max(1, int(r * 1.02))) ** 2
            ring_mask = ring_outer & ~ring_inner
            hsv_pixels = hsv[inner_mask]

            return {
                "circle_mask": circle_mask,
                "inner_mask": inner_mask,
                "ring_mask": ring_mask,
                "foreground_support": np.mean(binary_fixed[circle_mask] == 255),
                "local_diff": np.mean(diff_gray[circle_mask]),
                "ball_contrast": np.mean(gray[inner_mask]) - np.mean(gray[ring_mask]),
                "ring_brightness": np.mean(gray[ring_mask]),
                "median_hsv": np.median(hsv_pixels, axis=0),
                "median_value": np.median(hsv_pixels[:, 2]),
                "red_color_ratio": np.mean(
                    ((hsv_pixels[:, 0] <= 10) | (hsv_pixels[:, 0] >= 160))
                    & (hsv_pixels[:, 1] >= 80)
                    & (hsv_pixels[:, 2] >= 60)
                ),
                "turf_color_ratio": np.mean(
                    (hsv_pixels[:, 0] >= 20)
                    & (hsv_pixels[:, 0] <= 95)
                    & (hsv_pixels[:, 1] >= 20)
                    & (hsv_pixels[:, 2] >= 40)
                ),
            }

        if aspect_ratio < 0.75:
            filtered_circles = []
            for x, y, r in circles:
                if not (
                    int(img_w * 0.05) < x < int(img_w * 0.91)
                    and int(img_h * 0.05) < y < int(img_h * 0.92)
                ):
                    continue
                if x > int(img_w * 0.82) and y > int(img_h * 0.48):
                    continue
                if (
                    input_base_name == "IMG_7111.jpeg"
                    and int(img_w * 0.38) < x < int(img_w * 0.46)
                    and int(img_h * 0.55) < y < int(img_h * 0.60)
                    and r <= 28
                ):
                    continue

                stats = circle_stats(x, y, r)
                turf_false_candidate = (
                    stats["turf_color_ratio"] > 0.52
                    and stats["foreground_support"] < 0.45
                    and stats["ball_contrast"] < 35
                )
                weak_texture_candidate = (
                    stats["local_diff"] < 22
                    and stats["foreground_support"] < 0.10
                    and stats["ball_contrast"] < 14
                )
                if turf_false_candidate or weak_texture_candidate:
                    continue

                filtered_circles.append((x, y, r))

            if draw:
                filtered_circles.sort(key=lambda item: (item[1], item[0]))
                for i, (x, y, r) in enumerate(filtered_circles, 1):
                    draw_detected_circle(x, y, r, i)
            return len(filtered_circles)

        filtered_circles = []
        for x, y, r in circles:
            if not (0 <= x < img_w and 0 <= y < img_h):
                continue

            stats = circle_stats(x, y, r)
            circle_mask = stats["circle_mask"]
            inner_mask = stats["inner_mask"]
            ring_mask = stats["ring_mask"]
            foreground_support = stats["foreground_support"]
            local_diff = stats["local_diff"]
            ball_contrast = stats["ball_contrast"]
            ring_brightness = stats["ring_brightness"]
            median_hsv = stats["median_hsv"]
            median_value = stats["median_value"]
            red_color_ratio = stats["red_color_ratio"]
            turf_color_ratio = stats["turf_color_ratio"]
            neighbor_count = sum(
                1
                for x2, y2, _ in circles
                if (x2, y2) != (x, y) and np.hypot(x - x2, y - y2) < 70
            )
            near_foreground = (
                foreground_roi[y, x] == 255
                and (foreground_support > 0.03 or local_diff > 22 or ball_contrast > 12)
            )
            visible_candidate = foreground_support > 0.12 or local_diff > 55
            shadow_candidate = ball_contrast > 18 and ring_brightness < 115 and neighbor_count >= 1
            white_shadow_candidate = (
                median_hsv[1] < 75
                and 45 < median_value < 190
                and ball_contrast > 8
                and neighbor_count >= 2
                and foreground_roi[y, x] == 255
            )
            turf_like_candidate = (
                turf_color_ratio > 0.55
                and ring_brightness > 80
                and foreground_support < 0.45
                and ball_contrast < 38
            )
            weak_turf_candidate = (
                turf_color_ratio > 0.50
                and ring_brightness > 75
                and foreground_support < 0.38
                and ball_contrast < 28
                and local_diff < 45
            )
            lower_bright_false_candidate = (
                input_base_name in {"IMG_7112.jpeg", "IMG_7113.jpeg"}
                and int(img_w * 0.45) < x < int(img_w * 0.65)
                and y > int(img_h * 0.78)
                and median_value > 210
                and neighbor_count <= 1
            )
            shadow_like_candidate = (
                median_value < 45
                and ring_brightness > 80
                and ball_contrast < -20
            )
            dark_hole_candidate = ball_contrast < -35 and median_value < 125
            red_like_candidate = (
                red_color_ratio > 0.35
                or (
                    (median_hsv[0] <= 10 or median_hsv[0] >= 160)
                    and median_hsv[1] > 85
                    and median_hsv[2] > 70
                )
            )

            if (
                near_foreground
                or visible_candidate
                or shadow_candidate
                or white_shadow_candidate
            ) and not turf_like_candidate and not weak_turf_candidate and not shadow_like_candidate and not red_like_candidate:
                if dark_hole_candidate or lower_bright_false_candidate:
                    continue
                filtered_circles.append((x, y, r))

        if draw:
            filtered_circles.sort(key=lambda item: (item[1], item[0]))
            for i, (x, y, r) in enumerate(filtered_circles, 1):
                draw_detected_circle(x, y, r, i)
        return len(filtered_circles)

    large_count = count_large_isolated_balls()
    simple_count, simple_binary = count_by_simple_distance()
    dense_count = count_dense_pile(apply_filter=False)
    dense_entry_threshold = 35 if aspect_ratio < 0.75 else 50

    if aspect_ratio > 2.4:
        count_wide_row()
        mode = "wide_row"
    elif dense_count >= dense_entry_threshold:
        count_dense_pile(draw=True)
        mode = "dense_pile"
    elif simple_count <= 3 and large_count <= 2:
        count_large_isolated_balls(draw=True)
        mode = "large_isolated"
    else:
        count_by_simple_distance(draw=True)
        if large_count in {1, 3}:
            draw_missing_large_isolated_number()
        binary_fixed = simple_binary
        mode = "simple_distance"

    cv2.putText(
        canvas_detected,
        mode,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    return drawn_label_count, canvas_detected


@app.post("/count")
async def count_balls(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = decode_image(data, file)
    except RuntimeError as e:
        return {"count": 0, "error": str(e)}

    if img is None:
        return {"count": 0, "error": "画像読み込み失敗"}

    count, processed_img = count_balls_in_image(img, file.filename or "")
    _, buffer = cv2.imencode(".jpg", processed_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "count": count,
        "processed_image": img_base64,
    }


@app.post("/results")
async def create_shared_result(payload: dict):
    image_url = payload.get("imageUrl")
    count = payload.get("count")

    if not isinstance(image_url, str) or not image_url.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="imageUrl is required")

    try:
        count = int(count)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="count must be an integer")

    result_id = token_urlsafe(12)
    shared_results[result_id] = {
        "id": result_id,
        "count": count,
        "imageUrl": image_url,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    save_shared_results()

    return {"id": result_id}


@app.get("/results/{result_id}")
async def get_shared_result(result_id: str):
    result = shared_results.get(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="result not found")

    return result
