from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from io import BytesIO

try:
    from PIL import Image, ImageOps
    import pillow_heif
except ImportError:
    Image = None
    ImageOps = None
    pillow_heif = None

app = FastAPI()

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


@app.post("/count")
async def count_balls(file: UploadFile = File(...)):
    # 1. 画像データの読み込み
    data = await file.read()
    try:
        img = decode_image(data, file)
    except RuntimeError as e:
        return {"count": 0, "error": str(e)}

    if img is None:
        return {"count": 0, "error": "画像読み込み失敗"}

    # --- 提示されたコードのパラメータをそのまま使用 ---
    AREA_MIN, AREA_MAX = 100, 30000
    OUTER_AREA_MIN = 1000
    TARGET_WIDTH = 800
    THRESHOLD_VAL = 170

    # 2. リサイズ
    h, w = img.shape[:2]
    if w > TARGET_WIDTH:
        ratio = TARGET_WIDTH / w
        img = cv2.resize(img, (TARGET_WIDTH, int(h * ratio)))

    # --- 3. 提示された解析プロセス（そのまま実行） ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_raw = cv2.threshold(blurred, THRESHOLD_VAL, 255, cv2.THRESH_BINARY)

    # モルフォロジー演算
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    binary_fixed = cv2.morphologyEx(binary_raw, cv2.MORPH_OPEN, kernel3, iterations=2)
    binary_fixed = cv2.morphologyEx(binary_fixed, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # 距離変換と中心候補抽出
    dist_transform = cv2.distanceTransform(binary_fixed, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 輪郭の取得
    contours_centers, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_outer, _ = cv2.findContours(binary_fixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. カウントと描画処理
    count = 0
    counted_centers = []
    # 中心点の判定と番号描画
    for cnt in contours_centers:
        area = cv2.contourArea(cnt)
        if AREA_MIN < area < AREA_MAX:
            count += 1
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                counted_centers.append((cX, cY))
                # 青色で番号を描画
                cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
                cv2.putText(img, str(count), (cX-10, cY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 外枠の描画
    for cnt in contours_outer:
        has_counted_center = any(
            cv2.pointPolygonTest(cnt, center, False) >= 0
            for center in counted_centers
        )
        if has_counted_center and OUTER_AREA_MIN < cv2.contourArea(cnt) < AREA_MAX:
            cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)

    # 5. 画像をBase64文字列に変換
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # 【修正箇所】 React側とキー名・フォーマットを合わせる
    return {
        "count": count,
        "processed_image": img_base64
    }
