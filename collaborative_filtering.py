"""
Collaborative Filtering Recommender
=======================================
Recommends items based on patterns across *many users'* behavior,
rather than item attributes. Implements:

  1. User-based CF: find users with similar rating patterns to the
     target user, recommend what they liked.
  2. Item-based CF: find items rated similarly to items the target
     user already liked.
  3. Matrix factorization (SVD) via the `surprise`-style approach
     using simple NumPy SVD on the user-item matrix, as a lighter
     dependency-free alternative.

Author: <your name>
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringRecommender:
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """ratings_df columns: userId, movieId, rating
        movies_df columns: movieId, title, genres"""
        self.ratings = ratings_df
        self.movies = movies_df
        self._build_matrix()

    def _build_matrix(self):
        self.user_item_matrix = self.ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        ).fillna(0)

        self.user_similarity = pd.DataFrame(
            cosine_similarity(self.user_item_matrix),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

        self.item_similarity = pd.DataFrame(
            cosine_similarity(self.user_item_matrix.T),
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns,
        )

        self.movie_titles = self.movies.set_index("movieId")["title"].to_dict()

    # ------------------------------------------------------------------
    # User-based CF
    # ------------------------------------------------------------------
    def recommend_user_based(self, user_id: int, top_n: int = 5, k_neighbors: int = 10) -> pd.DataFrame:
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"user_id {user_id} not found.")

        similar_users = (
            self.user_similarity[user_id]
            .drop(user_id)
            .sort_values(ascending=False)
            .head(k_neighbors)
        )

        neighbor_ratings = self.user_item_matrix.loc[similar_users.index]
        weights = similar_users.values.reshape(-1, 1)

        weighted_sum = (neighbor_ratings.values * weights).sum(axis=0)
        weight_total = np.abs(weights).sum() + 1e-9
        predicted_scores = weighted_sum / weight_total

        predictions = pd.Series(predicted_scores, index=self.user_item_matrix.columns)

        already_rated = self.user_item_matrix.loc[user_id]
        predictions = predictions[already_rated == 0]  # exclude already-rated movies

        top = predictions.sort_values(ascending=False).head(top_n)
        return pd.DataFrame({
            "title": [self.movie_titles[m] for m in top.index],
            "predicted_score": top.values.round(3),
        })

    # ------------------------------------------------------------------
    # Item-based CF
    # ------------------------------------------------------------------
    def recommend_item_based(self, movie_title: str, top_n: int = 5) -> pd.DataFrame:
        title_to_id = self.movies.set_index("title")["movieId"].to_dict()
        if movie_title not in title_to_id:
            raise ValueError(f"'{movie_title}' not found in catalog.")

        movie_id = title_to_id[movie_title]
        if movie_id not in self.item_similarity.columns:
            raise ValueError(f"'{movie_title}' has no ratings yet to compare against.")

        similar = (
            self.item_similarity[movie_id]
            .drop(movie_id)
            .sort_values(ascending=False)
            .head(top_n)
        )
        return pd.DataFrame({
            "title": [self.movie_titles[m] for m in similar.index],
            "similarity": similar.values.round(3),
        })

    # ------------------------------------------------------------------
    # Matrix factorization (simple SVD-based latent factor model)
    # ------------------------------------------------------------------
    def recommend_svd(self, user_id: int, top_n: int = 5, n_factors: int = 10) -> pd.DataFrame:
        matrix = self.user_item_matrix.values
        user_means = matrix.mean(axis=1, keepdims=True)
        normalized = matrix - user_means

        U, sigma, Vt = np.linalg.svd(normalized, full_matrices=False)
        k = min(n_factors, len(sigma))
        U_k, sigma_k, Vt_k = U[:, :k], np.diag(sigma[:k]), Vt[:k, :]

        reconstructed = U_k @ sigma_k @ Vt_k + user_means

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        predicted = pd.Series(reconstructed[user_idx], index=self.user_item_matrix.columns)

        already_rated = self.user_item_matrix.loc[user_id]
        predicted = predicted[already_rated == 0]

        top = predicted.sort_values(ascending=False).head(top_n)
        return pd.DataFrame({
            "title": [self.movie_titles[m] for m in top.index],
            "predicted_score": top.values.round(3),
        })
