
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import cv2
import pandas as pd

from utils import get_center_box, get_position_foot


class Tracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        self.tracks = {
            "ball": [],
            "referee": [],
            "player": []
        }

    # ---------------------------------------------------------
    # DETECT FRAMES IN BATCHES
    # ---------------------------------------------------------
    def detect_frames(self, frames):

        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):

            print(
                f"Processing frames {i} - "
                f"{min(i + batch_size, len(frames))} "
                f"of {len(frames)}"
            )

            batch_frames = frames[i:i + batch_size]

            batch_detections = self.model.predict(
                batch_frames,
                conf=0.1,
                verbose=False
            )

            detections.extend(batch_detections)

        print("Total video frames:", len(frames))
        print("Total detections:", len(detections))

        return detections

    # ---------------------------------------------------------
    # INTERPOLATE MISSING BALL POSITIONS
    # ---------------------------------------------------------
    def calc_ball_position(self, ball_positions):

        if not ball_positions:
            return ball_positions

        boxes = [
            frame_data.get(1, {}).get("box", [np.nan] * 4)
            for frame_data in ball_positions
        ]

        df = pd.DataFrame(
            boxes,
            columns=["x1", "y1", "x2", "y2"],
            dtype=float
        )

        # If the ball was never detected, do not invent positions.
        if df.isna().all().all():
            return ball_positions

        # Interpolate between detected ball positions.
        df = df.interpolate(
            method="linear",
            limit_direction="both"
        )

        df = df.ffill().bfill()

        interpolated_positions = []

        for _, row in df.iterrows():

            box = row.tolist()

            if not np.isfinite(box).all():
                interpolated_positions.append({})
                continue

            interpolated_positions.append({
                1: {
                    "box": box
                }
            })

        return interpolated_positions

    # ---------------------------------------------------------
    # GET OBJECT TRACKS
    # ---------------------------------------------------------
    def get_object_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):

        # Load saved tracks if requested.
        if (
            read_from_stub
            and stub_path is not None
            and os.path.exists(stub_path)
        ):

            print("Loading tracks from stub...")

            with open(stub_path, "rb") as f:
                loaded_tracks = pickle.load(f)

            required_keys = {"ball", "player", "referee"}

            if (
                isinstance(loaded_tracks, dict)
                and required_keys.issubset(loaded_tracks.keys())
                and all(
                    len(loaded_tracks[key]) == len(frames)
                    for key in required_keys
                )
            ):

                self.tracks = loaded_tracks

                print("Tracks loaded successfully.")
                return self.tracks

            print("Stub is incompatible with this video.")
            print("Recalculating tracks...")

        # Reset tracking data.
        self.tracks = {
            "ball": [],
            "referee": [],
            "player": []
        }

        # Run YOLO detection.
        detections = self.detect_frames(frames)

        # Process each frame.
        for frame_num, detection in enumerate(detections):

            cls_names = detection.names

            cls_names_inv = {
                value: key
                for key, value in cls_names.items()
            }

            detection_supervision = (
                sv.Detections.from_ultralytics(detection)
            )

            # Convert goalkeeper detections to players.
            if (
                detection_supervision.class_id is not None
                and "goalkeeper" in cls_names_inv
                and "player" in cls_names_inv
            ):

                for object_ind, class_id in enumerate(
                    detection_supervision.class_id
                ):

                    if cls_names.get(int(class_id)) == "goalkeeper":

                        detection_supervision.class_id[
                            object_ind
                        ] = cls_names_inv["player"]

            # Update ByteTrack.
            detection_with_tracks = (
                self.tracker.update_with_detections(
                    detection_supervision
                )
            )

            # Create an entry for every frame.
            self.tracks["player"].append({})
            self.tracks["referee"].append({})
            self.tracks["ball"].append({})

            # Track players and referees.
            for frame_detection in detection_with_tracks:

                box = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id is None or track_id is None:
                    continue

                class_name = cls_names.get(int(cls_id))

                if class_name == "player":

                    self.tracks["player"][frame_num][
                        int(track_id)
                    ] = {
                        "box": box
                    }

                elif class_name == "referee":

                    self.tracks["referee"][frame_num][
                        int(track_id)
                    ] = {
                        "box": box
                    }

            # Detect the ball independently of ByteTrack.
            for frame_detection in detection_supervision:

                box = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id is None:
                    continue

                class_name = cls_names.get(int(cls_id))

                if class_name == "ball":

                    self.tracks["ball"][frame_num][1] = {
                        "box": box
                    }

        # Fill missing ball positions where possible.
        self.tracks["ball"] = self.calc_ball_position(
            self.tracks["ball"]
        )

        # Save tracking data if a stub path was provided.
        if stub_path is not None:

            directory = os.path.dirname(stub_path)

            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(stub_path, "wb") as f:
                pickle.dump(self.tracks, f)

            print("Tracks saved to:", stub_path)

        print("\nFinal track counts:")
        print("Player frames:", len(self.tracks["player"]))
        print("Referee frames:", len(self.tracks["referee"]))
        print("Ball frames:", len(self.tracks["ball"]))

        return self.tracks

    # ---------------------------------------------------------
    # ADD POSITIONS TO TRACKS
    # ---------------------------------------------------------
    def add_position_to_track(self, tracks):

        for object_name, object_tracks in tracks.items():

            for frame_num, frame_tracks in enumerate(object_tracks):

                for track_id, track_info in frame_tracks.items():

                    box = track_info.get("box")

                    if box is None:
                        continue

                    # Ball position is its center.
                    if object_name == "ball":
                        position = get_center_box(box)

                    # Player/referee position is at the feet.
                    else:
                        position = get_position_foot(box)

                    # IMPORTANT:
                    # frame_tracks is already the current frame's
                    # dictionary. Do not index it by object_name.
                    track_info["position"] = position
                    track_info["position_adjusted"] = position

    # ---------------------------------------------------------
    # DRAW ELLIPSE AND TRACK ID
    # ---------------------------------------------------------
    def draw_ellipse(
        self,
        frame,
        box,
        color,
        track_id=None
    ):

        x1, y1, x2, y2 = map(int, box)

        x_center = int((x1 + x2) / 2)
        width = max(1, x2 - x1)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(
                max(1, int(width)),
                max(1, int(0.35 * width))
            ),
            angle=0,
            startAngle=45,
            endAngle=255,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        if track_id is not None:

            rectangle_width = 40
            rectangle_height = 20

            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2

            y1_rect = y2 + 15
            y2_rect = y1_rect + rectangle_height

            cv2.rectangle(
                frame,
                (x1_rect, y1_rect),
                (x2_rect, y2_rect),
                color,
                cv2.FILLED
            )

            text = str(track_id)

            text_size = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )[0]

            text_x = x_center - text_size[0] // 2
            text_y = y1_rect + rectangle_height - 4

            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        return frame

    # ---------------------------------------------------------
    # DRAW TRIANGLE
    # ---------------------------------------------------------
    def draw_triangle(self, frame, box, color):

        x1, y1, x2, y2 = map(int, box)

        x_center = int((x1 + x2) / 2)

        triangle_points = np.array([
            [x_center, y1],
            [x_center - 10, y1 - 20],
            [x_center + 10, y1 - 20]
        ], dtype=np.int32)

        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            color,
            cv2.FILLED
        )

        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            (0, 0, 0),
            2
        )

        return frame

    # ---------------------------------------------------------
    # DRAW TEAM BALL CONTROL
    # ---------------------------------------------------------
    def draw_team_ball_control(
        self,
        frame,
        frame_num,
        team_ball_control
    ):

        overlay = frame.copy()

        height, width = frame.shape[:2]

        # Keep the overlay within the actual frame dimensions.
        x1 = max(0, width - 570)
        y1 = max(0, height - 130)
        x2 = width - 20
        y2 = height - 10

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.4,
            frame,
            0.6,
            0,
            frame
        )

        control = np.asarray(
            team_ball_control[:frame_num + 1]
        )

        team_1_count = np.count_nonzero(control == 1)
        team_2_count = np.count_nonzero(control == 2)

        total = team_1_count + team_2_count

        # Avoid division by zero when no team has possession.
        if total == 0:
            cv2.putText(
                frame,
                "Ball possession: unavailable",
                (x1 + 15, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2
            )
            return frame

        team_1 = team_1_count / total * 100
        team_2 = team_2_count / total * 100

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1:.2f}%",
            (x1 + 15, y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2:.2f}%",
            (x1 + 15, y1 + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2
        )

        return frame

    # ---------------------------------------------------------
    # DRAW ALL TRACKS ON VIDEO
    # ---------------------------------------------------------
    def draw_circle(
        self,
        video_frames,
        tracks,
        team_ball_control
    ):

        output_frames = []

        print("\nVideo frames:", len(video_frames))
        print("Player track frames:", len(tracks["player"]))
        print("Referee track frames:", len(tracks["referee"]))
        print("Ball track frames:", len(tracks["ball"]))

        # Use the common number of frames across all data.
        total_frames = min(
            len(video_frames),
            len(tracks["player"]),
            len(tracks["referee"]),
            len(tracks["ball"])
        )

        if total_frames != len(video_frames):
            print(
                "WARNING: Video and tracking frame counts differ."
            )

        for frame_num in range(total_frames):

            frame = video_frames[frame_num].copy()

            # -------------------------------------------------
            # DRAW PLAYERS
            # -------------------------------------------------
            player_dict = tracks["player"][frame_num]

            for track_id, player in player_dict.items():

                color = player.get(
                    "team_color",
                    (0, 0, 255)
                )

                frame = self.draw_ellipse(
                    frame,
                    player["box"],
                    color,
                    track_id
                )

                # Mark the player currently possessing the ball.
                if player.get("has_ball", False):

                    frame = self.draw_triangle(
                        frame,
                        player["box"],
                        (0, 0, 255)
                    )

            # -------------------------------------------------
            # DRAW REFEREES
            # -------------------------------------------------
            referee_dict = tracks["referee"][frame_num]

            for referee in referee_dict.values():

                frame = self.draw_ellipse(
                    frame,
                    referee["box"],
                    (0, 255, 0)
                )

            # -------------------------------------------------
            # DRAW BALL
            # -------------------------------------------------
            ball_dict = tracks["ball"][frame_num]

            for ball in ball_dict.values():

                box = ball.get("box", [])

                if (
                    len(box) == 4
                    and np.isfinite(box).all()
                ):

                    frame = self.draw_triangle(
                        frame,
                        box,
                        (255, 0, 0)
                    )

            # -------------------------------------------------
            # DRAW POSSESSION STATISTICS
            # -------------------------------------------------
            frame = self.draw_team_ball_control(
                frame,
                frame_num,
                team_ball_control
            )

            output_frames.append(frame)

        print("Output frames:", len(output_frames))

        return output_frames
