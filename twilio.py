import cv2
import numpy as np
import winsound
import pandas as pd
from datetime import datetime
from twilio.rest import Client

# Twilio credentials
account_sid = 'AC6c88b76c913a0fe8f5c2d629c7f8e28c'
auth_token = 'af8ac3d9d00a5b941b9427c9c98a09b5'
twilio_client = Client(account_sid, auth_token)
contact_number = '+12053869744'  # Replace with the actual contact number

# Load YOLO
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load classes
classes = []
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Get output layer names
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

colorRed = (0, 0, 255)
colorGreen = (0, 255, 0)

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or change to the appropriate index if you have multiple webcams

alarm_active = False
alarm_duration = 500
alarm_frequency = 1500

# Create a DataFrame to store object name and time data
df = pd.DataFrame(columns=['Object Name', 'Start Time', 'End Time'])

# Function to record object name and time
def record_object_time(object_name, start_time, end_time):
    global df
    df = pd.concat([df, pd.DataFrame({'Object Name': [object_name], 'Start Time': [start_time], 'End Time': [end_time]})],
                   ignore_index=True)

while True:
    ret, frame = cap.read()

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if more than one person is detected
    if sum(classes[class_ids[i]] == 'person' for i in indexes) > 1:
        # Send alert message
        message = twilio_client.messages.create(
            body='Multiple persons detected! Check the webcam feed.',
            from_='your_twilio_phone_number',
            to=contact_number
        )

    # Record object name and time if a person is detected
    alarm_active = any(classes[class_ids[i]] == 'person' for i in indexes)

    # ... (rest of the code remains unchanged)

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the recorded data to an Excel file
df.to_excel('D:/mm/object_time_records.xlsx', index=False)

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
