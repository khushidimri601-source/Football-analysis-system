from sklearn.cluster import KMeans


class Team_assign:

    def __init__(self):
        self.team_color = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):

        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=1
        ).fit(image_2d)

        return kmeans

    def get_player_color(self, frame, box):

        image = frame[
            int(box[1]):int(box[3]),
            int(box[0]):int(box[2])
        ]

        # Take only the upper half of the player
        half_image = image[
            0:int(image.shape[0] / 2),
            :
        ]

        # Cluster the pixels into 2 groups
        kmeans = self.get_clustering_model(half_image)

        labels = kmeans.labels_

        # Convert labels back into image shape
        clustered_image = labels.reshape(
            half_image.shape[0],
            half_image.shape[1]
        )

        # Get the clusters present in the four corners
        corner_cluster = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]

        # The cluster appearing most in corners
        # is assumed to be the background
        non_player_cluster = max(
            set(corner_cluster),
            key=corner_cluster.count
        )

        # The other cluster is the player's clothing
        player_cluster = 1 - non_player_cluster

        # Get RGB/BGR color of player's cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_teams(self, frame, player_detections):

        player_colors = []

        for _, player_detection in player_detections.items():

            box = player_detection['box']

            player_color = self.get_player_color(
                frame,
                box
            )

            player_colors.append(player_color)

        # Cluster players into two teams
        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=10
        ).fit(player_colors)

        self.kmeans = kmeans

        # Store team colors
        self.team_color[1] = kmeans.cluster_centers_[0]
        self.team_color[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_box, player_id):

        # If player was already assigned a team,
        # return the stored team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Get player's jersey color
        player_color = self.get_player_color(
            frame,
            player_box
        )

        # KMeans expects 2D input
        team_id = self.kmeans.predict(
            player_color.reshape(1, -1)
        )[0]

        # Convert:
        # 0 -> Team 1
        # 1 -> Team 2
        team_id += 1

        # Store team assignment
        self.player_team_dict[player_id] = team_id

        return team_id