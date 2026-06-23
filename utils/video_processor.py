import cv2
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    پردازش فایل‌های ویدیویی
    Video Processing Utilities
    """
    
    def __init__(self, video_path=None):
        self.video_path = video_path
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
    
    def open_video(self, video_path):
        """
        باز کردن فایل ویدیو
        """
        try:
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                logger.error(f"❌ نمی‌تونم ویدیو رو باز کنم: {video_path}")
                return False
            
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"✅ ویدیو باز شد: {video_path}")
            logger.info(f"   تعداد فریم: {self.frame_count}")
            logger.info(f"   FPS: {self.fps}")
            logger.info(f"   ابعاد: {self.width}x{self.height}")
            
            return True
        except Exception as e:
            logger.error(f"❌ خطا در باز کردن ویدیو: {e}")
            return False
    
    def read_frame(self):
        """
        خواندن فریم بعدی
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def get_frame_at(self, frame_number):
        """
        خواندن فریم خاص
        """
        if self.cap is None:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            return frame
        return None
    
    def resize_frame(self, frame, width=640, height=480):
        """
        تغییر اندازه فریم
        """
        return cv2.resize(frame, (width, height))
    
    def get_video_info(self):
        """
        دریافت اطلاعات ویدیو
        """
        return {
            'path': self.video_path,
            'frame_count': self.frame_count,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'duration_seconds': self.frame_count / self.fps if self.fps > 0 else 0
        }
    
    def close(self):
        """
        بستن ویدیو
        """
        if self.cap is not None:
            self.cap.release()
            logger.info("✅ ویدیو بسته شد")
    
    def __del__(self):
        self.close()


class WebcamProcessor:
    """
    پردازش تصاویر از وب‌کم
    Webcam Processing Utilities
    """
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.is_opened = False
    
    def open_webcam(self):
        """
        باز کردن وب‌کم
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error("❌ نمی‌تونم وب‌کم رو باز کنم")
                return False
            
            self.is_opened = True
            logger.info("✅ وب‌کم باز شد")
            return True
        except Exception as e:
            logger.error(f"❌ خطا: {e}")
            return False
    
    def read_frame(self):
        """
        خواندن فریم از وب‌کم
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def close(self):
        """
        بستن وب‌کم
        """
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            logger.info("✅ وب‌کم بسته شد")
    
    def __del__(self):
        self.close()
