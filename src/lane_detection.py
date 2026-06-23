import cv2
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LaneDetector:
    """
    تشخیص خطوط جاده
    Lane Detection using edge detection and Hough transform
    """
    
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
    
    def canny_edge_detection(self, image, threshold1=50, threshold2=150):
        """
        تشخیص لبه‌ها با Canny
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, threshold1, threshold2)
        return edges
    
    def region_of_interest(self, image, vertices):
        """
        انتخاب منطقه‌ی علاقه (ROI)
        """
        mask = np.zeros_like(image)
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    
    def hough_lines(self, image):
        """
        تشخیص خطوط با Hough Transform
        """
        lines = cv2.HoughLinesP(
            image,
            rho=2,
            theta=np.pi / 180,
            threshold=50,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=5
        )
        return lines
    
    def draw_lanes(self, image, lines, color=(0, 255, 0), thickness=3):
        """
        رسم خطوط جاده
        """
        output_image = image.copy()
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output_image, (x1, y1), (x2, y2), color, thickness)
        
        return output_image
    
    def detect_lanes(self, image):
        """
        تشخیص خطوط جاده کامل
        """
        height, width = image.shape[:2]
        
        # Canny edge detection
        edges = self.canny_edge_detection(image)
        
        # تعریف منطقه‌ی علاقه
        vertices = np.array([
            [(0, height),
             (width / 2, height / 2),
             (width, height)]
        ], dtype=np.int32)
        
        roi = self.region_of_interest(edges, vertices)
        
        # تشخیص خطوط
        lines = self.hough_lines(roi)
        
        return lines
    
    def draw_detected_lanes(self, image, lines):
        """
        رسم خطوط تشخیص‌شده روی تصویر
        """
        output_image = image.copy()
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # رنگ سبز برای خطوط
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # نقطه‌های مرجع
                cv2.circle(output_image, (x1, y1), 5, (0, 0, 255), -1)
                cv2.circle(output_image, (x2, y2), 5, (255, 0, 0), -1)
        
        return output_image
    
    def calculate_lane_deviation(self, image, lines):
        """
        محاسبه انحراف از خط جاده
        
        Returns:
            انحراف از مرکز (پیکسل)
        """
        if lines is None or len(lines) == 0:
            return None
        
        height, width = image.shape[:2]
        center_x = width / 2
        
        # محاسبه میانگین x برای خطوط
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-5)
            
            if slope < -0.5:  # خط چپ
                left_lines.append(line)
            elif slope > 0.5:  # خط راست
                right_lines.append(line)
        
        if len(left_lines) > 0 and len(right_lines) > 0:
            left_x = np.mean([line[0][0] for line in left_lines])
            right_x = np.mean([line[0][2] for line in right_lines])
            lane_center = (left_x + right_x) / 2
            
            deviation = center_x - lane_center
            return deviation
        
        return None
