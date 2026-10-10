
import pickle
import cv2
import numpy as np
import os


class CameraMovementEstimator:

    def __init__(self, frame):

        self.min_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03
            )
        )

        first_frame_gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )

        mask_features = np.zeros_like(first_frame_gray)

        # Select features away from the image edges.
        mask_features[:, 20:900] = 1
        mask_features[:, 1050:] = 1

        self.features = dict(
            maxCorners=200,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7,
            mask=mask_features
        )

    # ---------------------------------------------------------
    # ESTIMATE CAMERA MOVEMENT
    # ---------------------------------------------------------
    def get_camera_movement(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):

        if not frames:
            return []

        # Load cached movement if available.
        if (
            read_from_stub
            and stub_path is not None
            and os.path.exists(stub_path)
        ):

            with open(stub_path, "rb") as file:
                movement = pickle.load(file)

            if len(movement) == len(frames):
                return movement

            print("Camera movement stub length mismatch; recalculating.")

        camera_movement = [
            [0.0, 0.0] for _ in range(len(frames))
        ]

        old_gray = cv2.cvtColor(
            frames[0], cv2.COLOR_BGR2GRAY
        )

        old_features = cv2.goodFeaturesToTrack(
            old_gray,
            **self.features
        )

        for frame_num in range(1, len(frames)):

            frame_gray = cv2.cvtColor(
                frames[frame_num], cv2.COLOR_BGR2GRAY
            )

            if old_features is None or len(old_features) == 0:

                old_gray = frame_gray.copy()

                old_features = cv2.goodFeaturesToTrack(
                    old_gray,
                    **self.features
                )

                continue

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray,
                frame_gray,
                old_features,
                None,
                **self.lk_params
            )

            if new_features is not None and status is not None:

                valid = status.flatten() == 1

                good_new = new_features[valid].reshape(-1, 2)
                good_old = old_features[valid].reshape(-1, 2)

                if len(good_new) >= 3:

                    movements = good_new - good_old

                    # Remove extreme movements caused by independently
                    # moving objects or incorrect feature matches.
                    median_movement = np.median(
                        movements,
                        axis=0
                    )

                    deviations = np.linalg.norm(
                        movements - median_movement,
                        axis=1
                    )

                    inliers = deviations < 10

                    if np.count_nonzero(inliers) >= 3:

                        camera_movement_x, camera_movement_y = (
                            np.median(
                                movements[inliers],
                                axis=0
                            )
                        )

                        magnitude = np.hypot(
                            camera_movement_x,
                            camera_movement_y
                        )

                        if magnitude > self.min_distance:
                            camera_movement[frame_num] = [
                                float(camera_movement_x),
                                float(camera_movement_y)
                            ]

            # Refresh features for the next frame.
            old_gray = frame_gray.copy()

            old_features = cv2.goodFeaturesToTrack(
                frame_gray,
                **self.features
            )

        # Save calculated movement.
        if stub_path is not None:

            directory = os.path.dirname(
                os.path.abspath(stub_path)
            )

            os.makedirs(directory, exist_ok=True)

            with open(stub_path, "wb") as file:
                pickle.dump(camera_movement, file)

        return camera_movement

    # ---------------------------------------------------------
    # ADJUST TRACK POSITIONS
    # ---------------------------------------------------------
    def add_adjust_positions_to_tracks(
        self,
        tracks,
        camera_movement_per_frame
    ):

        for object_name, object_tracks in tracks.items():

            for frame_num, frame_tracks in enumerate(object_tracks):

                if frame_num >= len(camera_movement_per_frame):
                    break

                movement_x, movement_y = (
                    camera_movement_per_frame[frame_num]
                )

                for track_id, track_info in frame_tracks.items():

                    position = track_info.get("position")

                    if position is None:
                        continue

                    position_adjusted = (
                        position[0] - movement_x,
                        position[1] - movement_y
                    )

                    track_info["position_adjusted"] = (
                        position_adjusted
                    )

    # ---------------------------------------------------------
    # DRAW CAMERA MOVEMENT
    # ---------------------------------------------------------
    def draw_camera_movement(
        self,
        frames,
        camera_movement_per_frame
    ):

        output_frames = []

        for frame_num, frame in enumerate(frames):

            frame = frame.copy()
            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (500, 100),
                (255, 255, 255),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.6,
                frame,
                0.4,
                0
            )

            if frame_num < len(camera_movement_per_frame):

                movement_x, movement_y = (
                    camera_movement_per_frame[frame_num]
                )

            else:
                movement_x, movement_y = 0.0, 0.0

            cv2.putText(
                frame,
                f"Camera Movement X: {movement_x:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"Camera Movement Y: {movement_y:.2f}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

            output_frames.append(frame)

        return output_frames

    # Backward-compatible alias for your existing main.py.
    def add_adjust_position_to_track(
        self,
        tracks,
        camera_movement_per_frame
    ):

        return self.add_adjust_positions_to_tracks(
            tracks,
            camera_movement_per_frame
        )
