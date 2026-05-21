import mediapipe as mp
import cv2


class FaceDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh

        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        # Convert BGR → RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(
            rgb_frame
        )

        landmarks = None

        if results.multi_face_landmarks:
            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            h, w, _ = frame.shape
            landmarks = []

            for lm in face_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append((x, y))

        return frame, landmarks

    def draw_landmarks(
        self,
        frame,
        landmarks
    ):
        if landmarks:
            for (x, y) in landmarks:
                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

        return frame