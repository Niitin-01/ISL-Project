import cv2
from modules.video_capture import VideoCapture
from modules.face_detection import FaceDetector
from modules.expression import ExpressionDetector
from modules.head_movement import HeadMovementDetector
from modules.gaze import GazeDetector
from modules.eyebrow import EyebrowDetector
from modules.eye_state import EyeStateDetector
from utils.smoothing import TemporalSmoother
from modules.integration import ISLTextGenerator
import config


def main():
    video = VideoCapture()
    face_detector = FaceDetector()

    # Detectors
    expression_detector = ExpressionDetector()
    head_detector = HeadMovementDetector()
    gaze_detector = GazeDetector()
    eyebrow_detector = EyebrowDetector()
    eye_state_detector = EyeStateDetector()

    # Text generator
    text_generator = ISLTextGenerator()

    # Smoothers
    expression_smoother = TemporalSmoother(
        window_size=config.SMOOTHING_WINDOW
    )

    head_smoother = TemporalSmoother(
        window_size=config.SMOOTHING_WINDOW
    )

    gaze_smoother = TemporalSmoother(
        window_size=config.SMOOTHING_WINDOW
    )

    eyebrow_smoother = TemporalSmoother(
        window_size=config.SMOOTHING_WINDOW
    )

    eye_state_smoother = TemporalSmoother(
        window_size=config.FAST_SMOOTHING_WINDOW
    )

    tilt_smoother = TemporalSmoother(
        window_size=config.SMOOTHING_WINDOW
    )

    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 2

    # Cache predictions
    expression = "Neutral"
    head_movement = "Stable"
    gaze = "Center"
    eyebrow = "Neutral Brows"
    eye_state = "Eyes Open"
    head_tilt = "Straight"
    final_text = "Waiting for detection..."

    while True:
        frame = video.get_frame()

        if frame is None:
            break

        # Mirror flip
        frame = cv2.flip(frame, 1)

        # Resize
        frame = cv2.resize(
            frame,
            (640, 480)
        )

        # Image Denoising
        frame = cv2.GaussianBlur(
            frame,
            (5, 5),
            0
        )

        # Skip Alternate Frames
        frame_count += 1

        process_this_frame = (
            frame_count %
            PROCESS_EVERY_N_FRAMES
            == 0
        )

        landmarks = None

        if process_this_frame:
            frame, landmarks = (
                face_detector.process(
                    frame
                )
            )

            if landmarks:

                # Eyebrows
                raw_eyebrow = (
                    eyebrow_detector.detect(
                        landmarks
                    )
                )

                eyebrow = (
                    eyebrow_smoother.update(
                        raw_eyebrow
                    )
                )

                # Eye state
                raw_eye_state = (
                    eye_state_detector.detect(
                        landmarks
                    )
                )

                eye_state = (
                    eye_state_smoother.update(
                        raw_eye_state
                    )
                )

                # Expression
                raw_expression = (
                    expression_detector.detect(
                        landmarks,
                        eyebrow,
                        eye_state
                    )
                )

                expression = (
                    expression_smoother.update(
                        raw_expression
                    )
                )

                # Head movement
                raw_head = (
                    head_detector.detect(
                        landmarks
                    )
                )

                head_movement = (
                    head_smoother.update(
                        raw_head
                    )
                )

                # Gaze
                raw_gaze = (
                    gaze_detector.detect(
                        landmarks
                    )
                )

                gaze = (
                    gaze_smoother.update(
                        raw_gaze
                    )
                )

                # Head tilt
                raw_tilt = (
                    head_detector.detect_tilt(
                        landmarks
                    )
                )

                head_tilt = (
                    tilt_smoother.update(
                        raw_tilt
                    )
                )

                # Final sentence
                final_text = (
                    text_generator.generate(
                        expression,
                        head_movement,
                        gaze,
                        eyebrow=eyebrow,
                        eye_state=eye_state,
                        head_tilt=head_tilt
                    )
                )

            else:
                expression = "No Face"
                head_movement = "No Movement"
                gaze = "No Gaze"
                eyebrow = "No Eyebrow Cue"
                eye_state = "No Eye State"
                head_tilt = "No Tilt"

                final_text = (
                    "No person detected"
                )

                eyebrow_detector.reset()
                expression_detector.reset()

        # Draw landmarks
        frame = (
            face_detector.draw_landmarks(
                frame,
                landmarks
            )
        )

        # Display
        cv2.putText(
            frame,
            f"Expression: {expression}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Head: {head_movement}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Gaze: {gaze}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Brows: {eyebrow}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Eyes: {eye_state}",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Tilt: {head_tilt}",
            (20, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

        cv2.putText(
            frame,
            final_text,
            (20, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ISL Non-Manual Feature System",
            frame
        )

        # ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()


if __name__ == "__main__":
    main()