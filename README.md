# 🚗 Vehicle Vision System

سیستم تشخیص و تجزیه تصاویر برای خودروهای خودران با استفاده از YOLO v8 و Streamlit

## ✨ ویژگی‌ها

- 🛣️ **تشخیص خطوط جاده** - Lane Detection با دقت بالا
- 🚦 **شناسایی تابلوهای راهنمایی** - Traffic Sign Detection
- 🚗 **تشخیص اشیا** - Object Detection (خودرو، عابر پیاده، موانع)
- 📋 **شناسایی پلاک خودرو** - License Plate Detection
- 🔔 **سیستم هشدار** - Collision Warning و Lane Departure Alert
- 📹 **پردازش ویدیو** - Real-time video processing
- 🎨 **رابط کاربری** - Streamlit-based UI

## 📋 نیازمندی‌ها

- Python 3.8+
- YOLO v8 models:
  - `yolov8n.pt` (Nano - سریع)
  - `yolov8s.pt` (Small - متوازن)
  - `best.pt` (Custom trained)
  - `model.pt` (Custom model)
  - `license_plate.pt` (License plate detection)
- OpenCV
- PyTorch
- Streamlit
- NumPy

## 🚀 نصب و راه‌اندازی

### 1️⃣ Clone مخزن
```bash
git clone https://github.com/hannanehrahmani/vehicle-vision-system.git
cd vehicle-vision-system
```

### 2️⃣ ایجاد Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # روی Windows: venv\Scripts\activate
```

### 3️⃣ نصب Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ اضافه کردن Models
فایل‌های model رو در پوشه `models/` قرار بده:
```
models/
├── yolov8n.pt
├── yolov8s.pt
├── best.pt
├── model.pt
└── license_plate.pt
```

### 5️⃣ اجرای Streamlit App
```bash
streamlit run app.py
```

سپس مرورگر رو باز کن: **http://localhost:8501**

## 📁 ساختار پروژه

```
vehicle-vision-system/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py (Streamlit Application)
├── models/
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   ├── best.pt
│   ├── model.pt
│   └── license_plate.pt
├── src/
│   ├── __init__.py
│   ├── detector.py (YOLO Detection)
│   ├── lane_detection.py (Lane Detection)
│   ├── traffic_sign.py (Traffic Sign Recognition)
│   └── alerts.py (Alert System)
├── utils/
│   ├── __init__.py
│   ├── video_processor.py
│   └── visualization.py
└── data/
    ├── sample_video.mp4
    └── test_images/
```

## 🎯 کاربرد

1. **ورودی:** ویدیو یا تصویر از دوربین خودرو
2. **پردازش:** تشخیص خطوط، تابلوها، و اشیا
3. **خروجی:** ویدیو/تصویر پردازش‌شده با Bounding Boxes و هشدارها

## 🔧 مدل‌های استفاده‌شده

| مدل | توضیح | سرعت | دقت |
|-----|------|------|-----|
| **yolov8n.pt** | Nano - کم منابع | ⚡⚡⚡ | ⭐⭐ |
| **yolov8s.pt** | Small - متوازن | ⚡⚡ | ⭐⭐⭐ |
| **best.pt** | Custom trained | ⚡ | ⭐⭐⭐⭐ |
| **model.pt** | Custom model | ⚡ | ⭐⭐⭐⭐ |
| **license_plate.pt** | License plate | ⚡⚡ | ⭐⭐⭐ |

## ⚠️ سیستم هشدارها

- 🚨 **Lane Departure Alert** - هنگام خروج خودرو از خط
- 🚨 **Collision Warning** - هنگام نزدیک شدن بیش‌ازحد به خودروی جلویی
- 🚨 **Traffic Sign Warning** - تشخیص تابلوهای مهم (Stop, Speed Limit, etc.)
- 🚨 **Obstacle Detection** - تشخیص موانع در مسیر

## 📊 توابع اصلی

### 🎯 Detector (`src/detector.py`)
```python
from src.detector import ObjectDetector, LicensePlateDetector

# Initialize detector
detector = ObjectDetector(model_path="models/best.pt")

# Detect objects
results = detector.detect(image)

# Draw boxes
output = detector.draw_boxes(image, results)

# Get detection info
detections = detector.get_detections_info(results)
```

### 🛣️ Lane Detection (`src/lane_detection.py`)
```python
from src.lane_detection import LaneDetector

# Initialize lane detector
lane_detector = LaneDetector()

# Detect lanes
lanes = lane_detector.detect_lanes(image)

# Draw lanes
output = lane_detector.draw_detected_lanes(image, lanes)

# Calculate deviation
deviation = lane_detector.calculate_lane_deviation(image, lanes)
```

### 🚦 Traffic Sign Detection (`src/traffic_sign.py`)
```python
from src.traffic_sign import TrafficSignDetector

# Initialize traffic sign detector
traffic_detector = TrafficSignDetector()

# Detect signs
signs = traffic_detector.detect_signs(image)

# Draw signs
output = traffic_detector.draw_signs(image, signs)
```

### 🔔 Alert System (`src/alerts.py`)
```python
from src.alerts import AlertSystem

# Initialize alert system
alerts = AlertSystem()

# Check collision risk
alert = alerts.check_collision_risk(detections)

# Check lane departure
alert = alerts.check_lane_departure(lane_deviation)

# Get alert summary
summary = alerts.get_alert_summary()
```

## 🛠️ توسعه

برای اضافه کردن ویژگی‌های جدید:

1. فایل جدید در `src/` ایجاد کن
2. توابع رو implement کن
3. `app.py` رو آپدیت کن
4. Test کن و Commit کن

```bash
git add .
git commit -m "Add new feature: ..."
git push origin main
```

## 📝 License

MIT License

## 👨‍💻 نویسنده

**Hana** - Vehicle Vision System

## 🙋 کمک و مشارکت

برای گزارش مشکل یا پیشنهاد ویژگی جدید:
- Issue رو باز کن
- Pull Request رو ارسال کن
- ایمیل بفرست

---

**Made with ❤️ for Autonomous Vehicles**
