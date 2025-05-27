import cv2
import numpy as np
import math
from utils.draw_text_with_background import draw_text_with_background

def display_counter(frame, counter, position=(40, 240), color=(0, 0, 0), background_color=(192, 192, 192)):
    """Display the repetition counter."""
    text = f"Count: {counter}"
    draw_text_with_background(frame, text, position, 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, background_color, 1)

def display_stage(frame, stage, label="Stage", position=(40, 270), color=(0, 0, 0), background_color=(192, 192, 192)):
    """Display the current exercise stage."""
    text = f"{label}: {stage}"
    draw_text_with_background(frame, text, position, 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, background_color, 1)

def draw_progress_bar(frame, exercise, value, position, size=(200, 20), color=(0, 255, 0), background_color=(255, 255, 255)):
    """Draw a progress bar for tracking exercise repetitions."""
    x, y = position
    width, height = size
    
    # Get max value for the exercise type
    max_value = 10  # Default
    if exercise == "squat":
        max_value = 15
    elif exercise == "push_up":
        max_value = 10
    elif exercise == "hammer_curl":
        max_value = 12
    
    # Calculate fill width
    fill_width = int((value / max_value) * width)
    fill_width = min(fill_width, width)  # Ensure it doesn't exceed max width
    
    # Draw background
    cv2.rectangle(frame, (x, y), (x + width, y + height), background_color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), 1)
    
    # Draw fill
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
    
    # Draw text
    text = f"{value}/{max_value}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw label above the progress bar
    label = f"{exercise.replace('_', ' ').title()} Progress"
    draw_text_with_background(frame, label, (x, y - 10), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), (118, 29, 14), 1)

def draw_gauge_meter(frame, angle, text, position, radius=50, color=(0, 0, 255)):
    """Draw a gauge meter visualization showing the angle."""
    x, y = position
    start_angle = 180
    end_angle = 0
    
    # Draw outer circle
    cv2.circle(frame, (x, y), radius, (200, 200, 200), 2)
    
    # Calculate the angle position on the gauge
    gauge_angle = start_angle - (angle * (start_angle - end_angle) / 180)
    gauge_angle = max(min(gauge_angle, start_angle), end_angle) # Constrain angle
    
    # Convert to radians
    gauge_angle_rad = math.radians(gauge_angle)
    
    # Calculate point on circle
    gauge_x = int(x + radius * math.cos(gauge_angle_rad))
    gauge_y = int(y - radius * math.sin(gauge_angle_rad))
    
    # Draw line from center to angle point
    cv2.line(frame, (x, y), (gauge_x, gauge_y), color, 2)
    
    # Draw center circle
    cv2.circle(frame, (x, y), 5, color, -1)
    
    # Draw text
    cv2.putText(frame, f"{int(angle)}°", (x - 20, y + radius + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw title
    cv2.putText(frame, text, (x - radius, y - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
