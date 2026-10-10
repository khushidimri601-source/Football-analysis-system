
import numpy as np
import cv2


class ViewTransformer:

    def __init__(self):

        court_width = 68
        court_height = 23.32

        # Coordinates of the court in the original video
        self.pixel_vertices = np.array([
            [105, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ])

        # Corresponding coordinates on the transformed court
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_height, 0],
            [court_height, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Calculate perspective transformation matrix
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices,
            self.target_vertices
        )

    def transform_points(self, point):

        point = np.array(point)

        # Check whether the point lies inside the court polygon
        p = (int(point[0]), int(point[1]))

        is_inside = cv2.pointPolygonTest(
            self.pixel_vertices,
            p,
            False
        ) >= 0

        if not is_inside:
            return None

        # Reshape point for OpenCV perspective transformation
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)

        transformed_point = cv2.perspectiveTransform(
            reshaped_point,
            self.perspective_transformer
        )

        return transformed_point.reshape(-1, 2)

    def add_position_to_track(self, tracks):

        for object_name, object_tracks in tracks.items():

            for frame_num, frame_tracks in enumerate(object_tracks):

                for track_id, track_info in frame_tracks.items():

                    position = track_info.get("position_adjusted")

                    if position is None:
                        continue

                    position = np.array(position)

                    position_transformed = self.transform_points(position)

                    if position_transformed is not None:
                        position_transformed = (
                            position_transformed.squeeze().tolist()
                        )

                    tracks[object_name][frame_num][track_id][
                        "position_transformed"
                    ] = position_transformed


    