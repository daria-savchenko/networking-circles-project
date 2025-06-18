import cv2
import imutils
import numpy as np
from mylib import config
import math
import random

# Load model
net = cv2.dnn.readNetFromCaffe(
    "mobilenet_ssd/MobileNetSSD_deploy.prototxt",
    "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]

def generate_frames():
    cap = cv2.VideoCapture(config.url)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        # Start with the original frame
        output = frame.copy()

        # Create a dark overlay directly on top of it
        darkness = np.zeros_like(output)
        alpha = 0.9
        output = cv2.addWeighted(darkness, alpha, output, 1 - alpha, 0)

        # Detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                center_x = int((startX + endX) / 2)
                center_y = int((startY + endY) / 2)
                radius = int(max(endX - startX, endY - startY) / 2)

                # How many dots to draw around the circle
                dot_count = 100  # base number of dots per circle

                for layer in range(3):  # draw 3 layers of particles
                    ring_radius = radius + random.randint(-3, 3)  # slightly inner/outer rings
                    for j in range(dot_count):
                        angle = 2 * math.pi * j / dot_count
                        x = int(center_x + ring_radius * math.cos(angle))
                        y = int(center_y + ring_radius * math.sin(angle))

                        # Jitter for organic sparkle
                        jitter_x = random.randint(-2, 2)
                        jitter_y = random.randint(-2, 2)

                        # Vary size and color slightly
                        size = random.choice([1, 1, 2])
                        color = (
                            0,
                            random.randint(220, 255),  # greenish shimmer
                            random.randint(200, 255)
                        )

                        # Draw one sparkling dot
                        cv2.circle(output, (x + jitter_x, y + jitter_y), size, color, -1)

                label = f"Person: {int(confidence * 100)}%"
                cv2.putText(output, label, (center_x - radius, center_y - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', output)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
