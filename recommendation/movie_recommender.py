#!/usr/bin/env python3
"""
Movie Recommendation System
CODSOFT AI Internship - Task 4

This script implements a movie recommendation engine using two techniques:
1. Content-Based Filtering: Recommends movies similar to a target movie based on 
   genre and description metadata using TF-IDF vectorization and Cosine Similarity.
2. Collaborative Filtering: Suggests movies based on user-rating correlations.
"""

import sys
import pandas as pd
import numpy as np

# Verify scikit-learn is available before imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Error: The 'scikit-learn' library is required to run the recommendation system.")
    print("Please install it using: pip install scikit-learn pandas numpy")
    sys.exit(1)


class MovieRecommender:
    def __init__(self):
        # 1. Custom Dataset representing movies (genres and descriptions)
        self.movies_data = [
            {"id": 1, "title": "The Dark Knight", "genres": "Action Thriller Crime Drama", "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."},
            {"id": 2, "title": "Inception", "genres": "Action Sci-Fi Thriller", "description": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."},
            {"id": 3, "title": "Interstellar", "genres": "Sci-Fi Drama Adventure", "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival on another planet."},
            {"id": 4, "title": "The Matrix", "genres": "Action Sci-Fi", "description": "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth--the life he knows is the elaborate deception of an evil cyber-intelligence."},
            {"id": 5, "title": "Pulp Fiction", "genres": "Crime Drama", "description": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption."},
            {"id": 6, "title": "The Godfather", "genres": "Crime Drama", "description": "The aging patriarch of an organized crime dynasty in postwar New York City transfers control of his clandestine empire to his reluctant youngest son."},
            {"id": 7, "title": "Avengers: Endgame", "genres": "Action Sci-Fi Adventure", "description": "After the devastating events of Infinity War, the universe is in ruins. With the help of remaining allies, the Avengers assemble once more in order to reverse Thanos' actions and restore balance to the universe."},
            {"id": 8, "title": "Spider-Man: Into the Spider-Verse", "genres": "Action Animation Adventure Sci-Fi", "description": "Teen Miles Morales becomes the Spider-Man of his universe, and must join with five spider-powered individuals from other dimensions to stop a threat for all realities."},
            {"id": 9, "title": "Forrest Gump", "genres": "Drama Romance Comedy", "description": "The history of the United States from the 1950s to the 1970s unfolds from the perspective of an Alabama man with an IQ of 75, who yearns to be reunited with his childhood sweetheart."},
            {"id": 10, "title": "The Notebook", "genres": "Romance Drama", "description": "A poor and passionate young man falls in love with a rich young woman, giving her a sense of freedom, but they are soon separated because of their social differences."}
        ]
        
        self.movies_df = pd.DataFrame(self.movies_data)
        
        # 2. Custom User-Movie Ratings Matrix representing collaborative ratings
        # Rows: Users (User A, B, C, D, E), Columns: Movies (1 to 10)
        self.ratings_data = {
            "The Dark Knight": [5, 4, 1, 5, 2],
            "Inception":       [5, 5, 2, 4, 1],
            "Interstellar":    [4, 5, 1, 3, 1],
            "The Matrix":      [5, 4, 2, 5, 2],
            "Pulp Fiction":    [2, 1, 5, 4, 5],
            "The Godfather":   [3, 1, 5, 3, 5],
            "Avengers: Endgame": [4, 4, 1, 5, 1],
            "Spider-Man":      [5, 3, 1, 4, 2],
            "Forrest Gump":    [3, 2, 4, 3, 4],
            "The Notebook":    [1, 1, 4, 2, 5]
        }
        self.ratings_df = pd.DataFrame(self.ratings_data, index=["User A", "User B", "User C", "User D", "User E"])

    def get_content_recommendations(self, title, num_recommendations=3):
        """Content-Based Filtering using TF-IDF and Cosine Similarity."""
        # Clean title matching
        matching_movies = self.movies_df[self.movies_df['title'].str.lower() == title.lower()]
        if matching_movies.empty:
            return None
            
        target_idx = matching_movies.index[0]
        
        # Combine genres and descriptions for richer text feature extraction
        self.movies_df['metadata'] = self.movies_df['genres'] + " " + self.movies_df['description']
        
        # Initialize TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies_df['metadata'])
        
        # Calculate Cosine Similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Fetch similarity scores for the target movie index
        sim_scores = list(enumerate(cosine_sim[target_idx]))
        
        # Sort movies based on similarity scores in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Filter out the target movie itself and get top results
        sim_scores = [score for score in sim_scores if score[0] != target_idx]
        top_scores = sim_scores[:num_recommendations]
        
        # Retrieve movie metadata
        recommendations = []
        for idx, score in top_scores:
            recommendations.append({
                "title": self.movies_df.iloc[idx]['title'],
                "score": round(score, 3),
                "genres": self.movies_df.iloc[idx]['genres']
            })
            
        return recommendations

    def get_collaborative_recommendations(self, target_user, num_recommendations=3):
        """User-Based Collaborative Filtering using Pearson Correlation."""
        if target_user not in self.ratings_df.index:
            return None
            
        # Center ratings around each user's average rating (normalize ratings)
        normalized_ratings = self.ratings_df.sub(self.ratings_df.mean(axis=1), axis=0)
        
        # Compute Cosine Similarity between normalized users (similar to Pearson Correlation)
        user_sim = cosine_similarity(normalized_ratings)
        user_sim_df = pd.DataFrame(user_sim, index=self.ratings_df.index, columns=self.ratings_df.index)
        
        # Get similar users sorted
        similar_users = user_sim_df[target_user].sort_values(ascending=False)
        # Drop the target user themselves
        similar_users = similar_users.drop(target_user)
        
        # Identify movies the target user has rated lowly or average (e.g. rating <= 3) to recommend improvements,
        # or recommend movies they haven't rated (mocked as lowest rating in this dense matrix)
        target_user_ratings = self.ratings_df.loc[target_user]
        candidate_movies = target_user_ratings[target_user_ratings <= 3].index
        
        # Calculate weighted rating score predicted for candidate movies
        predictions = []
        for movie in candidate_movies:
            weighted_sum = 0
            similarity_sum = 0
            for peer, similarity in similar_users.items():
                if similarity <= 0:
                    continue
                peer_rating = self.ratings_df.loc[peer, movie]
                weighted_sum += similarity * peer_rating
                similarity_sum += similarity
                
            if similarity_sum > 0:
                pred_rating = weighted_sum / similarity_sum
                predictions.append({"movie": movie, "predicted_rating": round(pred_rating, 2)})
                
        # Sort and return recommendations
        predictions = sorted(predictions, key=lambda x: x['predicted_rating'], reverse=True)
        return predictions[:num_recommendations]


def main():
    recommender = MovieRecommender()
    print("=" * 60)
    print("      🎬 WELCOME TO THE MOVIE RECOMMENDATION SYSTEM 🎬")
    print("             Created for the CODSOFT AI Internship")
    print("=" * 60)
    
    while True:
        print("\nChoose Recommendation Technique:")
        print("1. Content-Based Filtering (Find similar movies)")
        print("2. Collaborative Filtering (Find recommendations for a User)")
        print("3. Exit\n")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            print("\nAvailable Movies in Database:")
            for idx, title in enumerate(recommender.movies_df['title'], 1):
                print(f" - {title}")
                
            print()
            movie_input = input("Enter a movie title: ").strip()
            recs = recommender.get_content_recommendations(movie_input)
            
            if recs is None:
                print(f"\nError: Movie '{movie_input}' was not found in the database. Please try again.")
            else:
                print(f"\n🎬 Top Content Recommendations for '{movie_input}':")
                print("-" * 50)
                for idx, rec in enumerate(recs, 1):
                    print(f"{idx}. {rec['title']} (Similarity Match: {rec['score']})")
                    print(f"   Genres: {rec['genres']}\n")
                    
        elif choice == "2":
            print("\nAvailable Users:")
            for user in recommender.ratings_df.index:
                print(f" - {user}")
                
            print()
            user_input = input("Enter a User name (e.g. User A): ").strip()
            recs = recommender.get_collaborative_recommendations(user_input)
            
            if recs is None:
                print(f"\nError: User '{user_input}' was not found.")
            else:
                print(f"\n👥 Recommended Movies for {user_input} (Based on Peer Preferences):")
                print("-" * 50)
                for idx, rec in enumerate(recs, 1):
                    print(f"{idx}. {rec['movie']} (Predicted Rating: {rec['predicted_rating']} / 5)")
                print()
                
        elif choice == "3":
            print("\nThank you for using the Recommendation System! Goodbye!")
            break
        else:
            print("\nInvalid selection! Please enter a number between 1 and 3.")


if __name__ == "__main__":
    main()
