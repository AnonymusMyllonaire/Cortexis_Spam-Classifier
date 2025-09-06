# 🎬 Movie Recommendation System – AI Internship Task 5

This project is a **basic movie recommendation system** built using Python. It leverages **collaborative filtering** based on user similarity and provides personalized movie suggestions.

---

## 📌 Objective

To demonstrate understanding and implementation of a **simple recommendation algorithm** using user-item interaction data.

---

## 🧠 Recommendation Logic

- **Approach**: User-based Collaborative Filtering
- **Similarity Metric**: Cosine Similarity
- **Algorithm**:
  1. Load MovieLens 100k dataset.
  2. Create a user-item rating matrix.
  3. Compute cosine similarity between users.
  4. Recommend movies liked by users with similar preferences.

---

## 📁 Dataset Used

- [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/100k/)
- Files used:
  - `u.data`: User ratings (`user_id`, `item_id`, `rating`, `timestamp`)
  - `u.item`: Movie metadata (`movie_id`, `title`, ...)

---

## 🛠️ Libraries Used

- `pandas`
- `numpy`
- `sklearn.metrics.pairwise` – Cosine similarity
- `matplotlib.pyplot` – Plotting
- `seaborn` – Heatmap visualization

---

## 🚀 How to Run

1. Make sure dataset files (`u.data` and `u.item`) are stored locally.
2. Install required Python packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn

Enter User ID (1–943): 5

Top 3 recommended movies for User 5:
1. Star Wars (1977)
2. The Empire Strikes Back (1980)
3. Return of the Jedi (1983)


python movie_recommender.py
