
import cv2

from utils import measure_distance, get_position_foot


class Speed_and_distance:

    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance(self, tracks):

        total_distance = {}

        for object_name, object_tracks in tracks.items():

            if object_name in ("ball", "referee"):
                continue

            total_distance[object_name] = {}

            number_of_frames = len(object_tracks)

            for frame_num in range(
                0, number_of_frames, self.frame_window
            ):

                last_frame = min(
                    frame_num + self.frame_window,
                    number_of_frames - 1
                )

                if last_frame == frame_num:
                    continue

                for track_id, _ in object_tracks[frame_num].items():

                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id].get(
                        "position_transformed"
                    )

                    end_position = object_tracks[last_frame][track_id].get(
                        "position_transformed"
                    )

                    if start_position is None or end_position is None:
                        continue

                    distance = measure_distance(
                        start_position,
                        end_position
                    )

                    time_elapsed = (
                        last_frame - frame_num
                    ) / self.frame_rate

                    if time_elapsed <= 0:
                        continue

                    speed_per_second = distance / time_elapsed
                    speed_per_hour = speed_per_second * 3.6

                    if track_id not in total_distance[object_name]:
                        total_distance[object_name][track_id] = 0.0

                    total_distance[object_name][track_id] += distance

                    for frame_num_batch in range(
                        frame_num, last_frame + 1
                    ):

                        if track_id not in object_tracks[frame_num_batch]:
                            continue

                        track_info = object_tracks[frame_num_batch][track_id]

                        track_info["speed"] = speed_per_hour
                        track_info["distance"] = (
                            total_distance[object_name][track_id]
                        )

    def draw_speed_and_distance(self, frames, tracks):

        object_frames = []

        for frame_num, frame in enumerate(frames):

            for object_name, object_tracks in tracks.items():

                if object_name in ("ball", "referee"):
                    continue

                if frame_num >= len(object_tracks):
                    continue

                for _, track_info in object_tracks[frame_num].items():

                    if "speed" not in track_info:
                        continue

                    speed = track_info.get("speed")
                    distance = track_info.get("distance")

                    if speed is None or distance is None:
                        continue

                    box = track_info["box"]

                    position = get_position_foot(box)
                    position = list(position)
                    position[1] += 40
                    position = tuple(map(int, position))

                    cv2.putText(
                        frame,
                        f"{speed:.2f} km/h",
                        position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{distance:.2f} m",
                        (position[0], position[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )

            object_frames.append(frame)

        return object_frames
