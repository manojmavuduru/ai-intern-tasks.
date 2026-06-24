import unittest
import pandas as pd

from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender


class TestContentBasedRecommender(unittest.TestCase):
    def setUp(self):
        self.movies = pd.DataFrame({
            "movieId": [1, 2, 3, 4],
            "title": ["Movie A", "Movie B", "Movie C", "Movie D"],
            "genres": ["Action|Sci-Fi", "Action|Sci-Fi", "Romance|Drama", "Comedy"],
        })
        self.recommender = ContentBasedRecommender(self.movies)

    def test_similar_to_returns_results(self):
        result = self.recommender.similar_to("Movie A", top_n=2)
        self.assertLessEqual(len(result), 2)
        self.assertNotIn("Movie A", result["title"].values)

    def test_similar_genre_movies_rank_higher(self):
        result = self.recommender.similar_to("Movie A", top_n=3)
        # Movie B shares identical genres with Movie A, should be top match
        self.assertEqual(result.iloc[0]["title"], "Movie B")

    def test_unknown_title_raises(self):
        with self.assertRaises(ValueError):
            self.recommender.similar_to("Nonexistent Movie")

    def test_recommend_for_user_history(self):
        result = self.recommender.recommend_for_user_history(["Movie A"], top_n=2)
        self.assertGreater(len(result), 0)


class TestCollaborativeFiltering(unittest.TestCase):
    def setUp(self):
        self.movies = pd.DataFrame({
            "movieId": [1, 2, 3, 4],
            "title": ["Movie A", "Movie B", "Movie C", "Movie D"],
            "genres": ["Action", "Action", "Romance", "Comedy"],
        })
        # Two clear taste clusters: users 1,2 like action; users 3,4 like romance
        self.ratings = pd.DataFrame({
            "userId": [1, 1, 2, 2, 3, 3, 4, 4],
            "movieId": [1, 2, 1, 2, 3, 4, 3, 4],
            "rating": [5, 5, 4, 5, 5, 4, 4, 5],
        })
        self.cf = CollaborativeFilteringRecommender(self.ratings, self.movies)

    def test_user_based_recommends_unrated_movies(self):
        result = self.cf.recommend_user_based(1, top_n=2)
        # User 1 already rated Movie A & B, so recommendations should be C/D
        self.assertTrue(all(t in ["Movie C", "Movie D"] for t in result["title"]))

    def test_item_based_finds_similar_movies(self):
        result = self.cf.recommend_item_based("Movie A", top_n=3)
        self.assertIn("Movie B", result["title"].values)

    def test_svd_runs_without_error(self):
        result = self.cf.recommend_svd(1, top_n=2)
        self.assertGreater(len(result), 0)

    def test_invalid_user_raises(self):
        with self.assertRaises(ValueError):
            self.cf.recommend_user_based(999)


if __name__ == "__main__":
    unittest.main()
