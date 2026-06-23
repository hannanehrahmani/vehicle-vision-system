import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    تشخیص اشیا با استفاده از YOLO v8
    Object Detection using YOLOv8
    """
    
    def __init__(self, model_path="models/best.pt", conf_threshold=0.5):
        """
        ایجاد detector
        
        Args:
            model_path: مسیر فایل مدل
            conf_threshold: حداقل confidence برای تشخیص
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.load_model()
        
    def load_model(self):
        """مدل را بارگذاری کن"""
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info(f"✅ مدل بارگذاری شد: {self.model_path}")
            else:
                logger.error(f"❌ مدل یافت نشد: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری مدل: {e}")
    
    def detect(self, image):
        """
        تشخیص اشیا در تصویر
        
        Args:
            image: تصویر ورودی (numpy array)
            
        Returns:
            نتایج تشخیص
        """
        if self.model is None:
            logger.error("❌ مدل بارگذاری نشده است")
            return None
        
        try:
            results = self.model(image, conf=self.conf_threshold)
            return results
        except Exception as e:
            logger.error(f"❌ خطا در تشخیص: {e}")
            return None
    
    def draw_boxes(self, image, results):
        """
        رسم Bounding Boxes روی تصویر
        
        Args:
            image: تصویر ورودی
            results: نتایج تشخیص
            
        Returns:
            تصویر با boxes کشیده شده
        """
        if results is None or len(results) == 0:
            return image
        
        output_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # دریافت مختصات
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # رنگ‌ها برای کلاس‌های مختلف
                colors = {
                    0: (0, 255, 0),      # Person - سبز
                    1: (255, 0, 0),      # Car - آبی
                    2: (0, 0, 255),      # Truck - قرمز
                    3: (255, 255, 0),    # Bus - سیان
                    4: (255, 0, 255),    # Motorcycle - منگنتا
                }
                
                color = colors.get(cls, (0, 255, 0))
                
                # رسم box
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                # نوشتن متن
                label = f"Class: {cls} Conf: {conf:.2f}"
                cv2.putText(output_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_image
    
    def get_detections_info(self, results):
        """
        دریافت اطلاعات تشخیص
        
        Returns:
            لیست دیکشنری‌های تشخیص‌شده
        """
        if results is None or len(results) == 0:
            return []
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detection = {
                    'class_id': cls,
                    'confidence': conf,
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'center_x': (x1 + x2) // 2,
                    'center_y': (y1 + y2) // 2
                }
                detections.append(detection)
        
        return detections


class LicensePlateDetector:
    """
    تشخیص پلاک خودرو
    License Plate Detection
    """
    
    def __init__(self, model_path="models/license_plate.pt"):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """بارگذاری مدل"""
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info(f"✅ مدل پلاک بارگذاری شد: {self.model_path}")
            else:
                logger.warning(f"⚠️ مدل پلاک یافت نشد: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ خطا: {e}")
    
    def detect(self, image):
        """تشخیص پلاک"""
        if self.model is None:
            return None
        
        try:
            results = self.model(image, conf=0.4)
            return results
        except Exception as e:
            logger.error(f"❌ خطا: {e}")
            return None
    
    def draw_plates(self, image, results):
        """رسم پلاک‌های تشخیص‌شده"""
        if results is None:
            return image
        
        output_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(output_image, "License Plate", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return output_image
