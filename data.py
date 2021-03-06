import pandas as pd
from sklearn import model_selection

UNWANTED_FEATURES = ["content_rating", "movie_imdb_link", "plot_keywords"]
CATEGORICAL_FEATURES = ["color", "country", "director_name", "language"]
NUMERICAL_FEATURES = ["actor_1_facebook_likes", "actor_2_facebook_likes", "actor_3_facebook_likes",
                      "aspect_ratio", "budget", "cast_total_facebook_likes",
                      "director_facebook_likes", "duration", "facenumber_in_poster", "gross",
                      "movie_facebook_likes", "num_critic_for_reviews", "num_user_for_reviews",
                      "num_voted_users", "title_year"]


class Data:

    def __init__(self, path):
        """
        Data class to handle csv files and pre process the data.
        :param path: Path to csv file.
        """
        self._data = pd.read_csv(path)
        self.scores = None  # binary data labels
        self.p_data = None  # processed data without labels

    def preprocess(self, threshold=7):
        """
        Preprocess the data including: converts categorial to dummies , normalizing and creates data labels.
        :param threshold: Parameter for score binarization.
        """
        # removes unwanted features , drops lines with empty data , removes duplicated "movie title" rows.
        self._data = self._data.drop(columns=UNWANTED_FEATURES).dropna(axis='rows').drop_duplicates(
            subset="movie_title", keep='first').drop(columns=["movie_title"])
        self._data.reset_index(inplace=True, drop=True)

        # creates dummy variables for categorical features.
        # separates "genres" values by pipeline "|".
        # merge all three actors together.
        dummy_genres = self._data.pop("genres").str.get_dummies(sep="|")
        self._data = pd.concat([self._data, dummy_genres], axis=1)
        actors = (self._data.pop("actor_1_name") + "|" + self._data.pop("actor_2_name") + "|" + self._data.pop(
            "actor_3_name")).str.get_dummies(sep="|")
        self._data = pd.concat([self._data, actors], axis=1)
        self._data = pd.get_dummies(self._data, columns=CATEGORICAL_FEATURES)

        # binarization of imdb scores by given threshold.
        self.scores = self._data.pop("imdb_score").values
        for i, score in enumerate(self.scores):
            if score >= threshold:
                self.scores[i] = 1
            else:
                self.scores[i] = 0

        # normalization of numerical features.
        self._data[NUMERICAL_FEATURES] = self._data[NUMERICAL_FEATURES].astype('float64')
        for feature in NUMERICAL_FEATURES:
            mean, std = self._data[feature].mean(), self._data[feature].std()
            for i in self._data.index:
                self._data.at[i, feature] = (self._data.at[i, feature] - mean) / std

        # processed data without labels
        self.p_data = self._data.values

    def split_to_k_folds(self, k=5):
        """
        Splits the data to a given K folds.
        :param k: Number of splits to initialize KFold
        :return: K folds of data (default k=5).
        """
        kf = model_selection.KFold(n_splits=k, shuffle=False, random_state=None)
        return kf.split(self.p_data)
