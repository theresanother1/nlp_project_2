# Topic Modeling on AG-News Dataset

This project explores various topic modeling techniques on the AG-News dataset, combining classical, matrix-based, and modern deep learning approaches. The goal is to discover meaningful topics in news articles and evaluate their alignment with the original AG-News categories.

## Project Structure

```
1_approach/
    implementation_approach_1.py
    use_implementation_1.ipynb
2_approach/
    implementation_approach_2.py
    use_implementation_2.ipynb
3_approach/
    implementation_approach_3.py
    use_implementation_3.ipynb
4_approach/
    implementation_approach_4.py
    use_implementation_4.ipynb
data/
    train.csv
    test.csv
    train_cleaned.csv
    test_cleaned.csv
requirements.txt
README.md
```

## Approaches

### 1. Classical Topic Modeling
- Implements LDA using `scikit-learn` and `gensim`.
- Hyperparameter tuning (number of topics, alpha, beta).
- Evaluates discovered topics against AG-News categories.
- See [`1_approach/use_implementation_1.ipynb`](1_approach/use_implementation_1.ipynb).

### 2. Matrix Factorization
- Applies Non-negative Matrix Factorization (NMF), Latent Semantic Analysis (LSA/LSI), and PCA.
- Compares TF-IDF and Bag-of-Words representations.
- Hyperparameter optimization for all matrix methods.
- See [`2_approach/use_implementation_2.ipynb`](2_approach/use_implementation_2.ipynb).

### 3. Deep Learning & Embeddings
- Uses BERTopic, Top2Vec.
- Evaluates transformer-based topic modeling.
- Tests ensemble methods.
- See [`3_approach/use_implementation_3.ipynb`](3_approach/use_implementation_3.ipynb).

### 4. Evaluation & Benchmarking
- Implements metrics: coherence, perplexity, topic diversity.
- Cross-validation and benchmarking of all models.
- Visualizes results and compares models.
- See [`4_approach/use_implementation_4.ipynb`](4_approach/use_implementation_4.ipynb).

## Data

- AG-News dataset: News articles with labels (`World`, `Sports`, `Business`, `Science/Tech`).
- Raw and cleaned CSV files in the `data/` directory.

## Getting Started

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
2. Run the notebooks in each approach folder to reproduce results.

## Results

- Each approach outputs discovered topics, evaluation metrics, and visualizations.
- Final benchmarking and recommendations are documented in the evaluation notebook.

## References

- [Topic Modeling – Vergleich verschiedener Methoden (Medium)](https://towardsdatascience.com/topic-modeling-with-lsa-plsa-lda-nmf-bertopic-top2vec-a-comparison-5e6ce4b1e4a5/)
- [GeeksForGeeks – Topic Modeling on News Articles](https://www.geeksforgeeks.org/topic-modeling-on-news-articles-using-unsupervised-ml/)
- [PhilippeHeitzmann Topic Modeling](https://philippeheitzmann.com/2022/02/topic-modeling-in-python/)
- [GitHub: Topic Modeling on News Articles](https://github.com/SarangGami/Topic-modeling-on-News-Articles-Unsupervised-Learning)

---

For details, see the notebooks in each approach folder.