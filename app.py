import json
import sys

import cv2
from datetime import datetime, timedelta
import requests

SLACK_TOKEN = ''

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)


def set_emoji(emoji: str):
    profile = {
        "status_text": "",
        "status_emoji": emoji
    }
    requests.get(f'https://slack.com/api/users.profile.set?token={SLACK_TOKEN}&profile={json.dumps(profile)}')


# When everything is done, release the capture

start = datetime.now()
away_time = timedelta(seconds=0)
print(f"{start} Start")
last_seen = datetime.now()
gone = False

while True:
    try:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # if len(faces) == 0 and last_seen is None:
        # print("Where did you go")
        # last_seen = datetime.now()

        if gone is False and len(faces) > 0:
            last_seen = datetime.now()

        if gone is False and datetime.now() - last_seen > timedelta(seconds=30):
            print(f"{datetime.now()} Bye!")
            set_emoji(":dash:")
            gone = True

        if gone is True and len(faces) > 0:
            print(f"{datetime.now()} Hello! You're back after {datetime.now() - last_seen} away")
            away_time += datetime.now() - last_seen
            print(
                f"{datetime.now()} Away: {away_time}\tPresent: {datetime.now() - start - away_time}\tTotal: {datetime.now() - start}")
            set_emoji(":male-technologist:")
            gone = False

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    except KeyboardInterrupt:
        video_capture.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
