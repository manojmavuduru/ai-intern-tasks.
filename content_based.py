"""
Content-Based Filtering Recommender
======================================
Recommends items similar to ones a user already liked, based on
item attributes (here: movie genres) rather than other users'
behavior. Uses TF-IDF vectorization of genre tags + cosine
similarity — a lightweight, interpretable approach that works even
for brand-new items with no ratings yet ("cold start" friendly).

Author: <your name>
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies_df: pd.DataFrame):
        """movies_df must have columns: movieId, title, genres (genres
        pipe-separated, e.g. 'Action|Sci-Fi')."""
        self.movies = movies_df.reset_index(drop=True)
        self._fit()

    def _fit(self):
        # Turn "Action|Sci-Fi" into "Action Sci-Fi" so TF-IDF treats
        # each genre as a token.
        genre_text = self.movies["genres"].str.replace("|", " ", regex=False)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(genre_text)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        self.title_to_index = {
            title: idx for idx, title in enumerate(self.movies["title"])
        }

    def similar_to(self, title: str, top_n: int = 5) -> pd.DataFrame:
        """Recommend movies similar to a given title based on genre overlap."""
        if title not in self.title_to_index:
            raise ValueError(f"'{title}' not found in catalog.")

        idx = self.title_to_index[title]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        result_idx = [i for i, _ in scores]
        result = self.movies.iloc[result_idx].copy()
        result["similarity"] = [round(s, 3) for _, s in scores]
        return result[["title", "genres", "similarity"]]

    def recommend_for_user_history(self, liked_titles: list, top_n: int = 5) -> pd.DataFrame:
        """Recommend movies based on the *average* profile of several
        movies a user has liked, rather than just one."""
        indices = [self.title_to_index[t] for t in liked_titles if t in self.title_to_index]
        if not indices:
            raise ValueError("None of the provided titles were found in the catalog.")

        user_profile = self.tfidf_matrix[indices].mean(axis=0)
        # cosine_similarity needs 2D array input
        import numpy as np
        user_profile = np.asarray(user_profile)
        scores = cosine_similarity(user_profile, self.tfidf_matrix)[0]

        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        scored = [s for s in scored if s[0] not in indices][:top_n]

        result_idx = [i for i, _ in scored]
        result = self.movies.iloc[result_idx].copy()
        result["similarity"] = [round(s, 3) for _, s in scored]
        return result[["title", "genres", "similarity"]]
