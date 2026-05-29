# Task 4: Recommendation System (Content & Collaborative)

This directory contains my implementation of a simple movie recommendation system using content-based filtering (TF-IDF vectorizer + Cosine Similarity) and user-based collaborative filtering (Pearson correlations).

## 🚀 How to Run

### **1. Command Line Interface (CLI)**
Run the recommender interactively in your terminal:
```bash
python movie_recommender.py
```

### **2. Jupyter Notebook**
Explore the linear algebra definitions, tf-idf formulas, and step-by-step vector matrix similarity outputs:
```bash
jupyter notebook recommendation_notebook.ipynb
```

---

## 🛠️ Implementation Details

### **1. Content-Based Filtering**
- **TF-IDF Vectorization**: Text metadata (combining genres and descriptions) is transformed into mathematical numerical term-frequency vectors.
- **Cosine Similarity**: We calculate the angles between the high-dimensional document vectors. Similar movies cluster close to each other.
- **Result**: When you search for `"The Dark Knight"`, it suggests action-thrillers with similar keyword contexts (e.g. `"The Matrix"` or `"Avengers: Endgame"`).

### **2. Collaborative Filtering**
- **User Correlation Matrix**: Normalized ratings represent individual biases (some users give high ratings generally, some give low). 
- **Weighted Prediction**: To predict User A's rating for a movie they haven't seen, we compute a weighted sum of their peers' ratings, weighted by how similar their past tastes are.
- **Result**: User A gets suggested movies that their peer group rated highly.
