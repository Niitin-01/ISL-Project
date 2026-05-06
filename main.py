import cv2
from modules.video_capture import VideoCapture
from modules.face_detection import FaceDetector
from modules.expression import ExpressionDetector
from modules.head_movement import HeadMovementDetector
from modules.gaze import GazeDetector
from utils.smoothing import TemporalSmoother
from modules.integration import ISLTextGenerator


def main():
    video = VideoCapture()
    face_detector = FaceDetector()

    # Detectors
    expression_detector = ExpressionDetector()
    head_detector = HeadMovementDetector()
    gaze_detector = GazeDetector()

    # Text generator (FINAL STEP)
    text_generator = ISLTextGenerator()

    # Smoothers
    expression_smoother = TemporalSmoother(window_size=10)
    head_smoother = TemporalSmoother(window_size=10)
    gaze_smoother = TemporalSmoother(window_size=10)

    while True:
        frame = video.get_frame()
        if frame is None:
            break

        # 🔥 Mirror flip
        frame = cv2.flip(frame, 1)

        # Resize
        frame = cv2.resize(frame, (640, 480))

        # Detect face
        frame, landmarks = face_detector.process(frame)

        if landmarks:
            # -------- Expression --------
            raw_expression = expression_detector.detect(landmarks)
            expression = expression_smoother.update(raw_expression)

            # -------- Head --------
            raw_head = head_detector.detect(landmarks)
            head_movement = head_smoother.update(raw_head)

            # -------- Gaze --------
            raw_gaze = gaze_detector.detect(landmarks)
            gaze = gaze_smoother.update(raw_gaze)

            # -------- FINAL TEXT --------
            final_text = text_generator.generate(expression, head_movement, gaze)

        else:
            expression = "No Face"
            head_movement = "No Movement"
            gaze = "No Gaze"
            final_text = "No person detected"

            # Reset smoothers
            expression_smoother = TemporalSmoother(window_size=10)
            head_smoother = TemporalSmoother(window_size=10)
            gaze_smoother = TemporalSmoother(window_size=10)

        # Draw landmarks
        frame = face_detector.draw_landmarks(frame, landmarks)

        # -------- Display --------
        cv2.putText(frame,
                    f"Expression: {expression}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.putText(frame,
                    f"Head: {head_movement}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2)

        cv2.putText(frame,
                    f"Gaze: {gaze}",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2)

        # -------- FINAL SENTENCE (IMPORTANT) --------
        cv2.putText(frame,
                    final_text,
                    (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

        # -------- Debug (optional) --------
        if landmarks:
            cv2.putText(frame,
                        f"Raw Exp: {raw_expression}",
                        (350, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

            cv2.putText(frame,
                        f"Raw Head: {raw_head}",
                        (350, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2)

            cv2.putText(frame,
                        f"Raw Gaze: {raw_gaze}",
                        (350, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

        cv2.imshow("ISL Non-Manual Feature System", frame)

        # Exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()


if __name__ == "__main__":
    main()