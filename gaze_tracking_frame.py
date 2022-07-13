import math
import cv2
from gaze_tracking import GazeTracking
import numpy as np
import time


def gaze_tracking_frame():
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    # Initial values for eye position tracking
    x_prev = y_prev = x_avg = y_avg = dist_prev = frame_count = duration = 0

    centers = [[], [], []]
    saved_centers = list()

    # Number of frames to be averaged together
    offset = 5
    start_viewing = time.time()

    # Check if the viewing time is more than 3 sec
    while time.time() - start_viewing < 20:
        # We get a new frame from the webcam
        ret, frame = webcam.read()

        if ret:
            # We send this frame to GazeTracking to analyze it
            gaze.refresh(frame)

            frame = gaze.annotated_frame()
            height, width = frame.shape[:2]

            # Pupil position in the eye
            text = ''
            if gaze.is_blinking():
                text = 'Blinking'
            elif gaze.is_right():
                text = 'Looking right'
            elif gaze.is_left():
                text = 'Looking left'
            elif gaze.is_center():
                text = 'Looking center'
            cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

            # Get the left eye and left pupil coordinates
            left_pupil = gaze.pupil_left_coords()
            left_eye = gaze.eye_left
            if frame_count < offset:
                if left_eye and left_pupil:
                    eye_left_left = left_eye.left
                    eye_left_right = left_eye.right
                    eye_left_top = left_eye.top
                    eye_left_bottom = left_eye.bottom

                    # Map the pupil position in the eye frame to the image frame
                    horizontal_ratio = left_eye.pupil.x / (left_eye.center[0] * 2 - 10)
                    vertical_ratio = left_eye.pupil.y / (left_eye.center[1] * 2 - 10)
                    x = int(horizontal_ratio * width)
                    y = int(vertical_ratio * height)

                    centers[0].append(x)
                    centers[1].append(y)
                    centers[2].append(time.time())

                    # Draw red box around the eye (eye frame)
                    cv2.rectangle(frame, (int(eye_left_left[0]), int(eye_left_top[1] - 10)),
                                  (int(eye_left_right[0]), int(eye_left_bottom[1] + 10)),
                                  (0, 0, 255), 2)
                    frame_count += 1

            # Calculate eye fixation duration
            if len(centers[0]) == offset:
                # Average the value of pupil center in 5 frames
                x_avg = np.mean(np.array(centers[0]))
                y_avg = np.mean(np.array(centers[1]))
                start = np.mean(np.array(centers[2]))

                # Calculate the pupil distance from the previous position
                dist = math.sqrt((x_avg - y_prev) ** 2 + (y_avg - x_prev) ** 2)
                diff = abs(dist - dist_prev)
                # print('dist =', dist)
                # print('diff =', diff)

                x_prev, y_prev = x_avg, y_avg
                dist_prev = dist

                # Check if the pupil position has changed
                if diff >= 50:
                    end = time.time()
                    # Calculate duration
                    duration = round(end - start, 4)
                    saved_centers.append([x_avg, y_avg, duration])
                    print('duration =', duration)

                frame_count = 0
                centers = [[], [], []]

            # Draw pupil position on the screen
            cv2.circle(frame, (int(x_avg), int(y_avg)), 5, (0, 0, 255), -1)
            cv2.putText(frame, 'Left pupil:  ' + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, 'Duration: ' + str(duration), (90, 195), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.imshow('frame', frame)

            time.sleep(1 / 30)

            if cv2.waitKey(1) == 27:
                break

    webcam.release()
    cv2.destroyAllWindows()

    return saved_centers


