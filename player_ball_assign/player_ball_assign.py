
import sys
sys.path.append('../')

from utils import get_center_box, measure_distance


class Player_ball_assign:

    def __init__(self):
        self.max_distance = 70

    def assign_ball_to_player(self, players, ball_box):

        ball_position = get_center_box(ball_box)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():

            player_box = player['box']

            distance_left = measure_distance(
                (player_box[0], player_box[-1]),
                ball_position
            )

            distance_right = measure_distance(
                (player_box[2], player_box[-1]),
                ball_position
            )

            distance = min(distance_left, distance_right)

            if distance < self.max_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player


    
