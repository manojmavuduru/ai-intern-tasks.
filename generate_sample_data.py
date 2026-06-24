"""
Sample Dataset Generator
==========================
Generates a small but realistic movies + ratings dataset so the
recommendation system runs out-of-the-box without needing an
external download. For a real project, swap this for MovieLens
(grouplens.org/datasets/movielens) — see README.

Author: <your name>
"""

import csv
import random
import os

MOVIES = [
    ("The Matrix", "Action|Sci-Fi"),
    ("Inception", "Action|Sci-Fi|Thriller"),
    ("The Notebook", "Romance|Drama"),
    ("Titanic", "Romance|Drama"),
    ("The Hangover", "Comedy"),
    ("Superbad", "Comedy"),
    ("Interstellar", "Sci-Fi|Drama"),
    ("Avengers: Endgame", "Action|Sci-Fi"),
    ("La La Land", "Romance|Musical"),
    ("The Conjuring", "Horror|Thriller"),
    ("Get Out", "Horror|Thriller"),
    ("Pride and Prejudice", "Romance|Drama"),
    ("John Wick", "Action|Thriller"),
    ("Mad Max: Fury Road", "Action|Sci-Fi"),
    ("Toy Story", "Animation|Comedy"),
    ("Finding Nemo", "Animation|Comedy"),
    ("The Shawshank Redemption", "Drama"),
    ("Forrest Gump", "Drama|Romance"),
    ("Parasite", "Thriller|Drama"),
    ("Knives Out", "Comedy|Thriller"),
    ("Coco", "Animation|Drama"),
    ("Up", "Animation|Drama"),
    ("It", "Horror"),
    ("A Quiet Place", "Horror|Thriller"),
    ("The Proposal", "Romance|Comedy"),
    ("Crazy Rich Asians", "Romance|Comedy"),
    ("Dune", "Sci-Fi|Drama"),
    ("Blade Runner 2049", "Sci-Fi|Thriller"),
    ("Whiplash", "Drama"),
    ("The Social Network", "Drama"),
]


def generate_dataset(num_users: int = 60, seed: int = 42, out_dir: str = "data"):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    movies_path = os.path.join(out_dir, "movies.csv")
    ratings_path = os.path.join(out_dir, "ratings.csv")

    with open(movies_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["movieId", "title", "genres"])
        for i, (title, genres) in enumerate(MOVIES, start=1):
            writer.writerow([i, title, genres])

    # Simulate users with "taste clusters" so collaborative filtering
    # has real signal to find (not pure noise).
    genre_lovers = {
        "action_fan": ["Action", "Sci-Fi", "Thriller"],
        "romantic": ["Romance", "Drama"],
        "comedy_fan": ["Comedy", "Animation"],
        "horror_fan": ["Horror", "Thriller"],
    }
    clusters = list(genre_lovers.keys())

    with open(ratings_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["userId", "movieId", "rating"])
        for user_id in range(1, num_users + 1):
            cluster = clusters[user_id % len(clusters)]
            liked_genres = genre_lovers[cluster]
            num_ratings = random.randint(8, 15)
            rated_movies = random.sample(range(1, len(MOVIES) + 1), num_ratings)

            for movie_id in rated_movies:
                _, genres = MOVIES[movie_id - 1]
                movie_genres = genres.split("|")
                matches_taste = any(g in liked_genres for g in movie_genres)
                if matches_taste:
                    rating = random.choice([4, 4, 5, 5, 5, 3])
                else:
                    rating = random.choice([1, 2, 2, 3, 3])
                writer.writerow([user_id, movie_id, rating])

    print(f"Generated {movies_path} ({len(MOVIES)} movies)")
    print(f"Generated {ratings_path} ({num_users} users)")


if __name__ == "__main__":
    generate_dataset()
