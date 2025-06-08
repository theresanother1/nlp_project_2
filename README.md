#  Topic Modeling –  Vorschlag Aufteilung

Vorschlag zur Arbeitsteilung im Projekt auf Basis des AG-News-Datensatzes. Fokus liegt auf der Kombination klassischer, matrixbasierter und moderner Deep-Learning-Ansätze für Topic Modelling.

---

## 1 – Klassisches Topic Modelling

- LDA mit `scikit-learn` & `gensim` auf AG-News implementieren und optimieren
- Hyperparameter-Tuning (Alpha, Beta, Anzahl Topics im Bereich 4–8)
- Vergleich unterschiedlicher Initialisierungsstrategien
- Performance der Topics gegen die 4 AG-News-Kategorien evaluieren

---

## 2 – Matrix-Faktorisierung

- Non-negative Matrix Factorization (NMF) auf AG-News anwenden
- Latent Semantic Analysis (LSA/LSI) mit `TruncatedSVD`
- PCA für Topic Discovery ausprobieren
- Performance-Vergleich: TF-IDF vs. Bag-of-Words
- Hyperparameter-Optimierung für alle Matrix-Methoden

---

## 3 – Deep Learning & Embeddings

- BERTopic mit AG-News Embeddings implementieren
- Top2Vec Integration testen
- Neural Topic Models (ProdLDA, AVITM) untersuchen
- Transformer-basierte Topic Modelling Ansätze evaluieren
- Ensemble-Methoden verschiedener Modelle testen

---

## 4 – Evaluation & Benchmarking

- Umsetzung verschiedener Metriken: Coherence, Perplexity, Topic Diversity
- Cross-Validation aller Modelle
- Laufzeit- & Ressourcen-Benchmarking
- Evaluation der Modelle gegen AG-News Labels: `World`, `Sports`, `Business`, `Sci/Tech`
- Visualisierungen der Ergebnisse erstellen
- Finaler Modellvergleich & Empfehlungen dokumentieren

---

## Ressourcen & Inspiration

- [Topic Modeling – Vergleich verschiedener Methoden (Medium)](https://towardsdatascience.com/topic-modeling-with-lsa-plsa-lda-nmf-bertopic-top2vec-a-comparison-5e6ce4b1e4a5/)
- [GeeksForGeeks – Topic Modeling auf News-Artikeln (1)](https://www.geeksforgeeks.org/topic-modeling-on-news-articles-using-unsupervised-ml/)
- [PhilippeHeitzmann Topic Modeling](https://philippeheitzmann.com/2022/02/topic-modeling-in-python/)
- [GitHub: Topic Modeling auf News-Artikeln](https://github.com/SarangGami/Topic-modeling-on-News-Articles-Unsupervised-Learning)

---

