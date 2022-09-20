import pandas as pd
import itertools
import numpy as np
import zipfile


class homogeneous:
    # Download data (quietly)
    tbl_player = "https://raw.githubusercontent.com/batuhan-demirci/fifa21_dataset/master/data/tbl_player.csv"
    tbl_player_skill = "https://raw.githubusercontent.com/batuhan-demirci/fifa21_dataset/master/data/tbl_player_skill.csv"
    tbl_team = "https://raw.githubusercontent.com/batuhan-demirci/fifa21_dataset/master/data/tbl_team.csv"

    # Load data
    player_df = pd.read_csv(tbl_player)
    skill_df = pd.read_csv(tbl_player_skill)
    team_df = pd.read_csv(tbl_team)

    # Extract subsets
    player_df = player_df[["int_player_id", "str_player_name",
                           "str_positions", "int_overall_rating", "int_team_id"]]
    skill_df = skill_df[["int_player_id", "int_long_passing",
                        "int_ball_control", "int_dribbling"]]
    team_df = team_df[["int_team_id", "str_team_name", "int_overall"]]

    # Merge data
    player_df = player_df.merge(skill_df, on='int_player_id')
    fifa_df = player_df.merge(team_df, on='int_team_id')

    # Sort dataframe
    fifa_df = fifa_df.sort_values(by="int_overall_rating", ascending=False)
    print("Players: ", fifa_df.shape[0])
    fifa_df.head()
    # Make sure that we have no duplicate nodes
    max(fifa_df["int_player_id"].value_counts())
    # Sort to define the order of nodes
    sorted_df = fifa_df.sort_values(by="int_player_id")
    # Select node features
    node_features = sorted_df[[
        "str_positions", "int_long_passing", "int_ball_control", "int_dribbling"]]
    # Convert non-numeric columns
    pd.set_option('mode.chained_assignment', None)
    positions = node_features["str_positions"].str.split(",", expand=True)
    node_features["first_position"] = positions[0]
    # One-hot encoding
    node_features = pd.concat([node_features, pd.get_dummies(
        node_features["first_position"])], axis=1, join='inner')
    node_features.drop(["str_positions", "first_position"],
                       axis=1, inplace=True)
    # Convert to numpy
    x = node_features.to_numpy()
    # Sort to define the order of nodes
    sorted_df = fifa_df.sort_values(by="int_player_id")
    # Select node features
    labels = sorted_df[["int_overall"]]
    # Convert to numpy
    y = labels.to_numpy()
    # Remap player IDs
    fifa_df["int_player_id"] = fifa_df.reset_index().index

    teams = fifa_df["str_team_name"].unique()
    all_edges = np.array([], dtype=np.int32).reshape((0, 2))
    for team in teams:
        team_df = fifa_df[fifa_df["str_team_name"] == team]
        players = team_df["int_player_id"].values
        # Build all combinations, as all players are connected
        permutations = list(itertools.combinations(players, 2))
        edges_source = [e[0] for e in permutations]
        edges_target = [e[1] for e in permutations]
        team_edges = np.column_stack([edges_source, edges_target])
        all_edges = np.vstack([all_edges, team_edges])
    # Convert to Pytorch Geometric format
    edge_index = all_edges.transpose()


class heterogeneous:
    anime_url = "https://raw.githubusercontent.com/Mayank-Bhatia/Anime-Recommender/master/data/anime.csv"
    rating_url = "https://raw.githubusercontent.com/Mayank-Bhatia/Anime-Recommender/master/data/rating.csv"
    anime = pd.read_csv(anime_url)
    rating = pd.read_csv(rating_url)
    # Sort to define the order of nodes
    sorted_df = anime.sort_values(by="anime_id").set_index("anime_id")

    # Map IDs to start from 0
    sorted_df = sorted_df.reset_index(drop=False)
    movie_id_mapping = sorted_df["anime_id"]

    # Select node features
    node_features = sorted_df[["type", "genre", "episodes"]]
    # Convert non-numeric columns
    pd.set_option('mode.chained_assignment', None)

    # For simplicity I'll just select the first genre here and ignore the others
    genres = node_features["genre"].str.split(",", expand=True)
    node_features["main_genre"] = genres[0]

    # One-hot encoding
    anime_node_features = pd.concat([node_features, pd.get_dummies(
        node_features["main_genre"])], axis=1, join='inner')
    anime_node_features = pd.concat([anime_node_features, pd.get_dummies(
        anime_node_features["type"])], axis=1, join='inner')
    anime_node_features.drop(["genre", "main_genre"], axis=1, inplace=True)
    # Convert to numpy
    x = anime_node_features.to_numpy()

    # Find out mean rating and number of ratings per user
    mean_rating = rating.groupby("user_id")["rating"].mean().rename("mean")
    num_rating = rating.groupby("user_id")["rating"].count().rename("count")
    user_node_features = pd.concat([mean_rating, num_rating], axis=1)

    # Remap user ID (to start at 0)
    user_node_features = user_node_features.reset_index(drop=False)
    user_id_mapping = user_node_features["user_id"]

    # Only keep features
    user_node_features = user_node_features[["mean", "count"]]
    # Convert to numpy
    x = user_node_features.to_numpy()
    # Movies that are part of our rating matrix
    rating["anime_id"].unique()
    # All movie IDs (e.g. no rating above for 1, 5, 6...)
    anime["anime_id"].sort_values().unique()
    # We can also see that there are some movies in the rating matrix, for which we have no features (we will drop them here)
    rating = rating[~rating["anime_id"].isin([30913, 30924, 20261])]
    # Extract labels
    labels = rating["rating"]
    # Convert to numpy
    y = labels.to_numpy()
    # Map anime IDs
    movie_map = movie_id_mapping.reset_index().set_index("anime_id").to_dict()
    rating["anime_id"] = rating["anime_id"].map(movie_map["index"]).astype(int)
    # Map user IDs
    user_map = user_id_mapping.reset_index().set_index("user_id").to_dict()
    rating["user_id"] = rating["user_id"].map(user_map["index"]).astype(int)
    edge_index = rating[["user_id", "anime_id"]].values.transpose()
