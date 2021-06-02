import pandas as pd
from surprise import Dataset, Reader

reader = Reader(rating_scale=(1, 10))


def get_dataset():
    anime = pd.read_parquet("datasets/anime.parquet")
    anime = anime[["anime_id", "type"]]
    users = pd.read_parquet("datasets/users.parquet")
    base_df = users.merge(anime, on="anime_id", how="left")
    return base_df


def filter_not_valid_types(base_df):
    base_df = base_df.dropna(subset=["type"])
    return base_df


def filter_most_frequently_type(base_df):
    unique_types = base_df["type"].unique()
    count_type_apperance = {
        i: int(base_df.loc[base_df["type"] == i][["type"]].count())
        for i in unique_types
    }
    sorted_types = sorted(count_type_apperance.items(), key=lambda x: x[1])
    return sorted_types[-1]


def get_data_from_most_frequent_type(base_df):
    anime_type = filter_most_frequently_type(base_df)
    base_df = base_df.loc[base_df["type"] == anime_type[0]]
    return base_df


def transform_dataframe_to_dataset(dataframe):
    dataframe = dataframe.rename(columns={"anime_id": "item_id"})
    base_dataset = Dataset.load_from_df(dataframe, reader)
    return base_dataset


def filter_data(base_df):
    base_df = filter_not_valid_types(base_df)
    base_df = get_data_from_most_frequent_type(base_df)
    return base_df


def split_data(base_df):
    base_df = filter_data(base_df)
    base_df = base_df[["user_id", "anime_id", "rating"]]
    dataset = transform_dataframe_to_dataset(base_df)
    return dataset


def filter_animes_without_grade(base_df):
    base_df = base_df.loc[~(base_df.rating == -1)]
    base_df = filter_data(base_df)
    base_df = base_df[["user_id", "anime_id", "rating"]]
    dataset = transform_dataframe_to_dataset(base_df)
    return dataset
