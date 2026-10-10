
from utils import read_video, save_video
from tracker import Tracker
from player_ball_assign import Player_ball_assign
from team_assigner import Team_assign
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance import Speed_and_distance

import cv2
import os
import numpy as np


def main():

    # --------------------------------------------------
    # File paths
    # --------------------------------------------------

    video_path = r'C:\Users\hp\Desktop\yolo\08fd33_4.mp4'
    model_path = r'model/best.pt'
    track_stub_path = r'C:\Users\hp\Desktop\yolo\stub\track_stubs.pkl'
    camera_stub_path = r'stub/camera_movement_stub.pkl'
    output_path = r'C:\Users\hp\Desktop\yolo\output_video\output.avi'

    # --------------------------------------------------
    # Read video
    # --------------------------------------------------

    video_frames = read_video(video_path)

    print("Number of video frames:", len(video_frames))

    if not video_frames:
        print("ERROR: No video frames were loaded.")
        return

    # --------------------------------------------------
    # Initialize tracker and detect objects
    # --------------------------------------------------

    tracker = Tracker(model_path)

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path=track_stub_path
    )

    # Add player and ball positions
    tracker.add_position_to_track(tracks)

    # Interpolate missing ball positions
    tracks["ball"] = tracker.calc_ball_position(tracks["ball"])

    # --------------------------------------------------
    # Camera movement estimation
    # --------------------------------------------------

    camera_movement_estimator = CameraMovementEstimator(
        video_frames[0]
    )

    camera_movement_per_frame = (
        camera_movement_estimator.get_camera_movement(
            video_frames,
            read_from_stub=True,
            stub_path=camera_stub_path
        )
    )

    camera_movement_estimator.add_adjust_position_to_track(
        tracks,
        camera_movement_per_frame
    )

    # --------------------------------------------------
    # Transform positions to pitch coordinates
    # --------------------------------------------------

    view_transformer = ViewTransformer()
    view_transformer.add_position_to_track(tracks)

    # --------------------------------------------------
    # Calculate player speed and distance
    # --------------------------------------------------

    speed_and_distance = Speed_and_distance()
    speed_and_distance.add_speed_and_distance(tracks)

    # --------------------------------------------------
    # Assign teams
    # --------------------------------------------------

    team_assigner = Team_assign()

    if tracks["player"] and tracks["player"][0]:

        team_assigner.assign_teams(
            video_frames[0],
            tracks["player"][0]
        )

    for frame_num, player_track in enumerate(tracks["player"]):

        for player_id, player in player_track.items():

            team = team_assigner.get_player_team(
                video_frames[frame_num],
                player["box"],
                player_id
            )

            player["team"] = team

            player["team_color"] = team_assigner.team_color.get(
                team,
                (0, 0, 255)
            )

    # --------------------------------------------------
    # Save a crop of the first detected player
    # --------------------------------------------------

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    for player in tracks["player"][0].values():

        x1, y1, x2, y2 = map(int, player["box"])

        frame = video_frames[0]
        height, width = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 > x1 and y2 > y1:

            cropped_image = frame[y1:y2, x1:x2]

            cv2.imwrite(
                os.path.join(output_dir, "cropped_image.jpg"),
                cropped_image
            )

        break

    # --------------------------------------------------
    # Assign ball possession
    # --------------------------------------------------

    player_assign = Player_ball_assign()
    team_ball_control = []

    previous_team = 0

    for frame_num, player_track in enumerate(tracks["player"]):

        # Reset possession flags
        for player in player_track.values():
            player["has_ball"] = False

        ball_dict = tracks["ball"][frame_num]
        ball_data = ball_dict.get(1)

        if ball_data:

            ball_box = ball_data["box"]

            assigned_player = player_assign.assign_ball_to_player(
                player_track,
                ball_box
            )

            if assigned_player in player_track:

                player_track[assigned_player]["has_ball"] = True

                previous_team = player_track[assigned_player].get(
                    "team", 0
                )

        # Keep exactly one possession entry per frame
        team_ball_control.append(previous_team)

    team_ball_control = np.array(team_ball_control)

    # --------------------------------------------------
    # Draw tracking results
    # --------------------------------------------------

    output_frames = tracker.draw_circle(
        video_frames,
        tracks,
        team_ball_control
    )

    if not output_frames:
        print("ERROR: No output frames were generated.")
        return

    # --------------------------------------------------
    # Draw camera movement information
    # --------------------------------------------------

    output_frames = camera_movement_estimator.draw_camera_movement(
        output_frames,
        camera_movement_per_frame
    )

    # --------------------------------------------------
    # Draw speed and distance information
    # --------------------------------------------------

    output_frames=speed_and_distance.draw_speed_and_distance(
        output_frames,
        tracks
    )

    # --------------------------------------------------
    # Save output video
    # --------------------------------------------------

    save_video(output_frames, output_path)

    print("Output video saved successfully:", output_path)


if __name__ == "__main__":
    main()
