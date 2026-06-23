import logging
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSystem:
    """
    سیستم هشدار برای خودروهای خودران
    Alert System for Autonomous Vehicles
    """
    
    def __init__(self, enable_sound=False):
        self.enable_sound = enable_sound
        self.alerts = []
        self.alert_log = []
    
    def check_collision_risk(self, detections, min_distance=100):
        """
        بررسی خطر تصادم
        
        Args:
            detections: لیست اشیای تشخیص‌شده
            min_distance: حداقل فاصله امن (پیکسل)
        
        Returns:
            Alert یا None
        """
        if detections is None or len(detections) == 0:
            return None
        
        # پیدا کردن خودرو در مرکز تصویر
        collision_risk_objects = []
        
        for detection in detections:
            # اگر شیء در بخش مرکزی تصویر باشد و نزدیک
            center_y = detection['center_y']
            height = detection['height']
            
            # اگر درفاصله کم باشد
            if height > 50:  # شیء نزدیک است
                collision_risk_objects.append(detection)
        
        if len(collision_risk_objects) > 0:
            alert = {
                'type': 'COLLISION_WARNING',
                'severity': 'HIGH',
                'message': '🚨 هشدار تصادم! فاصله خودرو جلویی خیلی کم است!',
                'timestamp': datetime.now(),
                'data': collision_risk_objects
            }
            self.alerts.append(alert)
            self.log_alert(alert)
            return alert
        
        return None
    
    def check_lane_departure(self, lane_deviation, threshold=50):
        """
        بررسی خروج از خط جاده
        
        Args:
            lane_deviation: انحراف از خط (پیکسل)
            threshold: حداقل انحراف برای هشدار
        
        Returns:
            Alert یا None
        """
        if lane_deviation is None:
            return None
        
        if abs(lane_deviation) > threshold:
            direction = "چپ" if lane_deviation < 0 else "راست"
            alert = {
                'type': 'LANE_DEPARTURE',
                'severity': 'MEDIUM',
                'message': f'⚠️ خروج از خط جاده! انحراف به {direction}: {abs(lane_deviation):.1f}px',
                'timestamp': datetime.now(),
                'deviation': lane_deviation,
                'direction': direction
            }
            self.alerts.append(alert)
            self.log_alert(alert)
            return alert
        
        return None
    
    def check_traffic_sign(self, signs, important_signs=None):
        """
        بررسی تابلوهای مهم
        
        Args:
            signs: لیست تابلوهای تشخیص‌شده
            important_signs: لیست تابلوهای مهم
        
        Returns:
            Alert یا None
        """
        if important_signs is None:
            important_signs = ['STOP', 'SPEED_LIMIT', 'YIELD', 'NO_ENTRY']
        
        if signs is None or len(signs) == 0:
            return None
        
        for sign in signs:
            if sign['type'] in important_signs:
                alert = {
                    'type': 'TRAFFIC_SIGN_ALERT',
                    'severity': 'MEDIUM',
                    'message': f"🚦 تابلو {sign['type']} تشخیص داده شد!",
                    'timestamp': datetime.now(),
                    'sign_type': sign['type'],
                    'sign_color': sign['color']
                }
                self.alerts.append(alert)
                self.log_alert(alert)
                return alert
        
        return None
    
    def check_obstacle(self, detections, obstacle_classes=None):
        """
        بررسی وجود موانع
        
        Args:
            detections: لیست اشیای تشخیص‌شده
            obstacle_classes: کلاس‌های مختلف موانع
        
        Returns:
            Alert یا None
        """
        if obstacle_classes is None:
            obstacle_classes = ['person', 'bicycle', 'motorcycle']
        
        if detections is None or len(detections) == 0:
            return None
        
        for detection in detections:
            # بررسی اگر مانع در مسیر باشد
            if detection['height'] > 30:  # مانع نزدیک است
                alert = {
                    'type': 'OBSTACLE_ALERT',
                    'severity': 'HIGH',
                    'message': '🚨 مانع در مسیر تشخیص داده شد!',
                    'timestamp': datetime.now(),
                    'obstacle': detection
                }
                self.alerts.append(alert)
                self.log_alert(alert)
                return alert
        
        return None
    
    def log_alert(self, alert):
        """
        ثبت هشدار در لاگ
        """
        self.alert_log.append(alert)
        logger.warning(f"🚨 {alert['message']}")
    
    def get_latest_alerts(self, count=5):
        """
        دریافت آخرین هشدارها
        """
        return self.alerts[-count:] if len(self.alerts) > 0 else []
    
    def clear_alerts(self):
        """
        پاک کردن هشدارها
        """
        self.alerts = []
    
    def get_alert_summary(self):
        """
        دریافت خلاصه هشدارها
        """
        summary = {
            'total_alerts': len(self.alert_log),
            'collision_warnings': len([a for a in self.alert_log if a['type'] == 'COLLISION_WARNING']),
            'lane_departures': len([a for a in self.alert_log if a['type'] == 'LANE_DEPARTURE']),
            'traffic_signs': len([a for a in self.alert_log if a['type'] == 'TRAFFIC_SIGN_ALERT']),
            'obstacles': len([a for a in self.alert_log if a['type'] == 'OBSTACLE_ALERT']),
        }
        return summary
