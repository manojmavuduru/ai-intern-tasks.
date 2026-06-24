"""
CLI Demo for the Recommendation System
=========================================
Runs all three recommendation strategies (content-based, user-based
CF, item-based CF, SVD) on the sample dataset and prints results.

Usage:
    python demo.py
"""

import os
import pandas as pd

from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender


def main():
    data_dir = "data"
    if not os.path.exists(os.path.join(data_dir, "movies.csv")):
        print("Sample data not found — generating it now...")
        from generate_sample_data import generate_dataset
        generate_dataset()

    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))

    print("=" * 60)
    print("CONTENT-BASED FILTERING")
    print("=" * 60)
    cb = ContentBasedRecommender(movies)
    print("\nBecause you watched 'The Matrix':")
    print(cb.similar_to("The Matrix", top_n=5))

    print("\nBased on liking The Matrix + Inception + Interstellar:")
    print(cb.recommend_for_user_history(["The Matrix", "Inception", "Interstellar"], top_n=5))

    print("\n" + "=" * 60)
    print("COLLABORATIVE FILTERING")
    print("=" * 60)
    cf = CollaborativeFilteringRecommender(ratings, movies)

    sample_user = ratings["userId"].iloc[0]
    print(f"\nUser-based CF recommendations for user {sample_user}:")
    print(cf.recommend_user_based(sample_user, top_n=5))

    print(f"\nItem-based CF — movies similar to 'John Wick' (based on rating patterns):")
    print(cf.recommend_item_based("John Wick", top_n=5))

    print(f"\nSVD (latent factor) recommendations for user {sample_user}:")
    print(cf.recommend_svd(sample_user, top_n=5))


if __name__ == "__main__":
    main()
