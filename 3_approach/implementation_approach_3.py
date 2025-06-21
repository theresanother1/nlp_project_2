# AG-News Topic Modeling Study
# Comprehensive evaluation of modern topic modeling approaches
import os
import umap
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import kagglehub
import pandas as pd
import numpy as np
import nltk
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
from data_pipeline import data_processing
warnings.filterwarnings('ignore')

# Topic modeling libraries
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    BERTOPIC_AVAILABLE = True
except ImportError:
    print("BERTopic not available. Install with: pip install bertopic")
    BERTOPIC_AVAILABLE = False

try:
    from top2vec import Top2Vec

    TOP2VEC_AVAILABLE = True
except ImportError:
    print("Top2Vec not available. Install with: pip install top2vec")
    TOP2VEC_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available for neural topic models")
    PYTORCH_AVAILABLE = False


class AGNewsTopicModeling:
    """
    Comprehensive topic modeling study on AG-News dataset
    Implements multiple approaches: BERTopic, Top2Vec, Neural Topic Models, and Ensemble methods
    """

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.processed_texts = []
        self.labels = []
        self.test_texts = []
        self.test_labels = []
        self.results = {}

        # AG-News categories
        self.category_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

    def _preprocess_text(self, texts):
        processed_texts = []
        valid_indices = []
        #print("PREPROCESSING TEXTS...")
        for i, text in enumerate(texts):
            #rint("before PREPROCESS")
            #print(texts)
            text = str(text).lower()

            # Remove special characters and numbers (if still there)
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', ' ', text)

            text = text.strip()
            text = text.split()
            #print("BEFORE STOPWORDS")
            #print(text)
            #print(stopwords.words('english'))
            text = [word for word in text if len(word) > 2]

            #if len(text) > 10:  # Only keep texts with meaningful content
            processed_texts.append(" ".join(text))
        return processed_texts, valid_indices



    def load_and_preprocess_data(self, train_data, test_data, text_column='Combined', label_column='labels'):
        """
        Load and preprocess AG-News data with given preprocessing pipeline
        """
        print("Loading and preprocessing AG-News data...")
        #self.initialize_nltk()

        # Extract texts and labels
        texts = train_data[text_column].tolist()
        print("LABELS are:")

        print(set(train_data[label_column].tolist()))
        self.labels = train_data[label_column].tolist()
        processed_texts, _ = self._preprocess_text(texts)

        test_texts = test_data[text_column].tolist()
        self.test_labels = test_data[label_column].tolist()
        processed_test_texts, _ = self._preprocess_text(test_texts)

        self.processed_texts = processed_texts
        self.test_texts = processed_test_texts
        #print("Train")
        #print(processed_texts)
        #print("Test")
        #print(processed_test_texts)

        print(f"Loaded {len(self.processed_texts)} training texts and {len(self.test_texts)} test texts")
        print(f"Categories distribution: {Counter(self.labels)}")

    def find_optimal_clusters_elbow(self, embeddings, max_k=15):
        """
        Find optimal number of clusters using Elbow method and SSE
        """
        print("Finding optimal number of clusters using Elbow method...")

        sse = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            sse.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(embeddings, cluster_labels))

        # Find elbow point
        # Calculate the rate of change
        deltas = np.diff(sse)
        second_deltas = np.diff(deltas)
        elbow_k = k_range[np.argmax(second_deltas) + 2]  # +2 due to double diff

        # Also consider silhouette score
        best_silhouette_k = k_range[np.argmax(silhouette_scores)]

        print(f"Elbow method suggests k = {elbow_k}")
        print(f"Best silhouette score at k = {best_silhouette_k}")
        print(f"Ground truth has k = 4 categories")

        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(k_range, sse, 'bo-')
        ax1.axvline(x=elbow_k, color='r', linestyle='--')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Sum of Squared Errors (SSE)')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True)

        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.axvline(x=best_silhouette_k, color='r', linestyle='--')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("elbow_vs_silhouette.png", dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="white",)
        plt.show()


        return elbow_k, best_silhouette_k, sse, silhouette_scores

    def evaluate_topic_model(self, predicted_topics, true_labels, model_name):
        """
        Evaluate topic model performance against ground truth labels
        """
        #print("PREDICTED")
        #print(set(predicted_topics))
        #print("TRUE")
        #print(set(true_labels))
        # Convert to numpy arrays
        predicted_topics = np.array(predicted_topics)
        true_labels = np.array(true_labels)

        # Calculate clustering metrics
        ari = adjusted_rand_score(true_labels, predicted_topics)
        nmi = normalized_mutual_info_score(true_labels, predicted_topics)
        homogeneity = homogeneity_score(true_labels, predicted_topics)
        completeness = completeness_score(true_labels, predicted_topics)
        v_measure = v_measure_score(true_labels, predicted_topics)

        metrics = {
            'ARI': ari,
            'NMI': nmi,
            'Homogeneity': homogeneity,
            'Completeness': completeness,
            'V-Measure': v_measure
        }

        print(f"\n{model_name} Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        return metrics


    def run_bertopic(self, n_topics=4):
        """
        Implement BERTopic with AG-News embeddings
        """
        if not BERTOPIC_AVAILABLE:
            print("BERTopic not available. Skipping...")
            return None, None

        print("\n=== Implementing BERTopic ===")

        model_name = 'all-MiniLM-L6-v2'
        # Use sentence transformer for embeddings
        embedding_model = SentenceTransformer(model_name)

        # Generate embeddings for optimal k finding
        print("Generating embeddings for optimal k calculation...")
        embeddings = embedding_model.encode(self.processed_texts)

        # Find optimal k
        elbow_k, silhouette_k, _, _ = self.find_optimal_clusters_elbow(embeddings)

        # Initialize BERTopic with optimal number of topics
        topic_model = BERTopic(
            nr_topics=n_topics,
            min_topic_size=5,
            embedding_model=embedding_model,
            verbose=True
        )

        # Fit the model
        print("Fitting BERTopic model...")
        topics, probs = topic_model.fit_transform(self.processed_texts)

        # Test on test set
        test_topics, test_probs = topic_model.transform(self.test_texts)

        # Evaluate performance
        train_metrics = self.evaluate_topic_model(topics, self.labels, f"BERTopic (Train) - {model_name}")
        test_metrics = self.evaluate_topic_model(test_topics, self.test_labels, f"BERTopic (Test) - {model_name}")

        # Store results
        self.results['BERTopic'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model': topic_model,
            'train_topics': topics,
            'test_topics': test_topics
        }

        # UMAP-Reduktion auf 2D
        umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine')
        reduced_embeddings = umap_model.fit_transform(embeddings)
        reduced_embeddings_embeddings_test = umap_model.transform(embedding_model.encode(self.test_texts))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot mit echten Labels
        axes[0].scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=self.labels, cmap='tab10', s=10)
        axes[0].set_title("UMAP Projection (Ground Truth) - Train")
        axes[1].scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=topics, cmap='tab10', s=10)
        axes[1].set_title("UMAP Projection (BERTopic Topics) - Train")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot mit echten Labels
        axes[0].scatter(reduced_embeddings_embeddings_test[:, 0], reduced_embeddings_embeddings_test[:, 1], c=self.test_labels, cmap='tab10', s=10)
        axes[0].set_title("UMAP Projection (Ground Truth) - Train")
        axes[1].scatter(reduced_embeddings_embeddings_test[:, 0], reduced_embeddings_embeddings_test[:, 1], c=test_topics, cmap='tab10', s=10)
        axes[1].set_title("UMAP Projection (BERTopic Topics) - Train")
        plt.show()


        # Display topic information
        print("\nTop words per topic:")
        topic_info = topic_model.get_topic_info()
        print(topic_info.head(10))

        return topic_model, topics

    def run_top2vec(self):
        """
        Implement and test Top2Vec integration
        """
        if not TOP2VEC_AVAILABLE:
            print("Top2Vec not available. Skipping...")
            return None, None

        print("\n=== Implementing Top2Vec ===")

        # Initialize Top2Vec
        print("Training Top2Vec model...")
        model = Top2Vec(
            documents=self.processed_texts,
            speed="learn",  # Options: learn, fast-learn, deep-learn
            workers=4,
            min_count=5
        )

        print(f"Top2Vec found {model.get_num_topics()} topics")

        train_topics = []
        # Use get_documents_topics directly
        try:
            for i in range(len(self.processed_texts)):
                doc_topics, doc_scores = model.get_documents_topics([i])
                if len(doc_topics) > 0:
                    train_topics.append(doc_topics[0])
                else:
                    train_topics.append(-1)
        except Exception as e:
            print(f"Error getting document topics directly: {e}")
            # Fallback using search_documents_by_documents
            try:
                for i in range(len(self.processed_texts)):
                    # Top2Vec search_documents_by_documents returns 3 values, not 4
                    topic_nums, topic_scores, doc_scores = model.search_documents_by_documents([i], num_docs=1)
                    if len(topic_nums) > 0:
                        train_topics.append(topic_nums[0])
                    else:
                        train_topics.append(-1)
            except Exception as e2:
                print(f"Error with search method: {e2}")
                # Final fallback - assign random topics
                num_topics = model.get_num_topics()
                train_topics = [i % num_topics for i in range(len(self.processed_texts))]


        # For test set, we need to infer topics
        test_topics = []
        for test_doc in self.test_texts:
            try:
                # Search for similar documents
                topic_nums, topic_scores, doc_scores, doc_ids = model.search_documents_by_keywords([test_doc],
                                                                                                   num_docs=1)
                if len(topic_nums) > 0:
                    test_topics.append(topic_nums[0])
                else:
                    test_topics.append(-1)
            except:
                test_topics.append(-1)

        # Evaluate performance
        train_metrics = self.evaluate_topic_model(train_topics, self.labels, "Top2Vec (Train)")
        test_metrics = self.evaluate_topic_model(test_topics, self.test_labels, "Top2Vec (Test)")

        # Store results
        self.results['Top2Vec'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model': model,
            'train_topics': train_topics,
            'test_topics': test_topics
        }

        print(train_topics)


        # Display topic words
        print("\nTop words per topic:")
        print("GET NUM TOPICS")
        print(model.get_num_topics())
        topics = model.get_topics()


        for i in range(min(model.get_num_topics(), 8)):
            # print("TOPICS top words are ")
            print(topics[0][i])

            print(f"Topic {i}: {', '.join(topics[0][i][:10])}")

        return model, train_topics

    def run_neural_topic_model(self, n_topics=4, epochs=100):
        """
        Implement Neural Topic Models using OCTIS ProdLDA
        """
        try:
            from octis.models.ProdLDA import ProdLDA
            from octis.dataset.dataset import Dataset
        except ImportError:
            print("OCTIS not available. Installing OCTIS is required for ProdLDA...")
            print("Please install with: pip install octis")
            return self.implement_lda_baseline(n_topics)

        print("\n=== Implementing Neural Topic Model (OCTIS ProdLDA) ===")

        # Prepare data in OCTIS format
        # OCTIS expects documents as lists of tokens
        train_docs = [text.split() for text in self.processed_texts]
        test_docs = [text.split() for text in self.test_texts]

        # Create vocabulary from training data
        vocab = set()
        for doc in train_docs:
            vocab.update(doc)
        vocab = list(vocab)

        # Create OCTIS dataset
        dataset = Dataset()

        # Manually set the dataset attributes
        dataset.corpus = train_docs
        dataset.vocabulary = vocab
        dataset._Dataset__metadata = {}

        # Set partitions manually
        n_train = len(train_docs)
        dataset._Dataset__metadata["last-training-doc"] = n_train - 1
        dataset._Dataset__metadata["last-validation-doc"] = n_train - 1

        # Initialize ProdLDA model
        model = ProdLDA(
            num_topics=n_topics,
            num_epochs=epochs,
            learn_priors=True,
            batch_size=64,
            lr=0.001,
            momentum=0.99,
            solver='adam',
            num_samples=20,
            reduce_on_plateau=True,
            num_layers=2,
            num_neurons=100,
            activation='softplus',
            dropout=0.2,
            use_partitions=True
        )

        print("Training OCTIS ProdLDA model...")

        # Train the model
        model_output = model.train_model(dataset)

        # Get topic assignments for training data
        train_topics = []
        for doc in train_docs:
            # Get topic distribution for document
            doc_topic_dist = model.inference([doc])
            # Get dominant topic
            dominant_topic = np.argmax(doc_topic_dist[0])
            train_topics.append(dominant_topic)

        train_topics = np.array(train_topics)

        # Get topic assignments for test data
        test_topics = []
        for doc in test_docs:
            try:
                doc_topic_dist = model.inference([doc])
                dominant_topic = np.argmax(doc_topic_dist[0])
                test_topics.append(dominant_topic)
            except:
                # If inference fails for a document, assign random topic
                test_topics.append(np.random.randint(0, n_topics))

        test_topics = np.array(test_topics)

        # Evaluate performance
        train_metrics = self.evaluate_topic_model(train_topics, self.labels, "Neural Topic Model (Train)")
        test_metrics = self.evaluate_topic_model(test_topics, self.test_labels, "Neural Topic Model (Test)")

        # Store results
        self.results['Neural_TM'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model': model,
            'train_topics': train_topics,
            'test_topics': test_topics,
            'model_output': model_output
        }

        # Display top words per topic
        print("\nTop words per topic:")
        try:
            # Get topics from model output
            topics = model_output['topics']
            for i, topic_words in enumerate(topics):
                # Take top 10 words for each topic
                top_words = topic_words[:10] if len(topic_words) >= 10 else topic_words
                print(f"Topic {i}: {', '.join(top_words)}")
        except Exception as e:
            print(f"Could not display topics: {e}")


        return model, train_topics

    def run_transformer_topic_modeling(self):
        """
        Implement transformer-based topic modeling approaches
        """
        print("\n=== Implementing Transformer-based Topic Modeling ===")

        # This is essentially BERTopic with different configurations
        if not BERTOPIC_AVAILABLE:
            print("BERTopic not available for transformer-based approach. Skipping...")
            return None, None

        # Try different transformer models
        transformer_models = [
            #'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'distilbert-base-nli-mean-tokens'
        ]

        best_model = None
        best_score = -1

        for model_name in transformer_models:
            try:
                print(f"Testing {model_name}...")
                embedding_model = SentenceTransformer(model_name)
                # Generate embeddings for optimal k finding
                print("Generating embeddings for optimal k calculation...")
                embeddings = embedding_model.encode(self.processed_texts)

                # Find optimal k
                elbow_k, silhouette_k, _, _ = self.find_optimal_clusters_elbow(embeddings)

                topic_model = BERTopic(
                    nr_topics=elbow_k,
                    embedding_model=embedding_model,
                    verbose=False
                )

                topics, _ = topic_model.fit_transform(self.processed_texts)
                test_topics, _ = topic_model.transform(self.test_texts)

                # UMAP-Reduktion auf 2D
                umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine')
                reduced_embeddings = umap_model.fit_transform(embeddings)
                reduced_embeddings_embeddings_test = umap_model.transform(embedding_model.encode(self.test_texts))

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                # Plot mit echten Labels
                axes[0].scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=self.labels, cmap='tab10', s=10)
                axes[0].set_title(f"UMAP Projection (Ground Truth) - Train")
                axes[1].scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=topics, cmap='tab10', s=10)
                axes[1].set_title(f"UMAP Projection (BERTopic Topics - {model_name}) - Train")
                plt.show()

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                # Plot mit echten Labels
                axes[0].scatter(reduced_embeddings_embeddings_test[:, 0], reduced_embeddings_embeddings_test[:, 1],
                                c=self.test_labels, cmap='tab10', s=10)
                axes[0].set_title("UMAP Projection (Ground Truth) - Test")
                axes[1].scatter(reduced_embeddings_embeddings_test[:, 0], reduced_embeddings_embeddings_test[:, 1],
                                c=test_topics, cmap='tab10', s=10)
                axes[1].set_title(f"UMAP Projection (BERTopic Topics {model_name}) - Test")
                plt.show()

                # Quick evaluation
                test_nmi = normalized_mutual_info_score(self.test_labels, test_topics)
                print(f"{model_name} Test NMI: {test_nmi:.4f}")

                if test_nmi > best_score:
                    best_score = test_nmi
                    best_model = topic_model
                    best_train_topics = topics
                    best_test_topics = test_topics
                    best_model_name = model_name

            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue

        if best_model is not None:
            print(f"\nBest transformer model: {best_model_name}")
            train_metrics = self.evaluate_topic_model(best_train_topics, self.labels, f"Transformer-TM (Train)")
            test_metrics = self.evaluate_topic_model(best_test_topics, self.test_labels, f"Transformer-TM (Test)")

            self.results['Transformer_TM'] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'model': best_model,
                'train_topics': best_train_topics,
                'test_topics': best_test_topics,
                'best_model_name': best_model_name
            }

        return best_model, best_train_topics

    def implement_ensemble_methods(self):
        """
        Test ensemble methods combining different topic models
        """
        print("\n=== Implementing Ensemble Methods ===")

        if len(self.results) < 2:
            print("Need at least 2 models for ensemble. Skipping...")
            return None

        # Collect predictions from all available models
        train_predictions = []
        test_predictions = []
        model_names = []

        for model_name, result in self.results.items():
            if 'train_topics' in result and 'test_topics' in result:
                train_predictions.append(result['train_topics'])
                test_predictions.append(result['test_topics'])
                model_names.append(model_name)

        if len(train_predictions) < 2:
            print("Insufficient models with predictions for ensemble.")
            return None

        print(f"Ensembling {len(train_predictions)} models: {model_names}")

        # Voting ensemble
        train_predictions = np.array(train_predictions).T  # Shape: (n_samples, n_models)
        test_predictions = np.array(test_predictions).T

        # Majority vote for each sample
        train_ensemble = []
        test_ensemble = []

        for i in range(len(train_predictions)):
            # Get most common prediction
            vote_counts = Counter(train_predictions[i])
            majority_vote = vote_counts.most_common(1)[0][0]
            train_ensemble.append(majority_vote)

        for i in range(len(test_predictions)):
            vote_counts = Counter(test_predictions[i])
            majority_vote = vote_counts.most_common(1)[0][0]
            test_ensemble.append(majority_vote)

        # Evaluate ensemble performance
        train_metrics = self.evaluate_topic_model(train_ensemble, self.labels, "Ensemble (Train)")
        test_metrics = self.evaluate_topic_model(test_ensemble, self.test_labels, "Ensemble (Test)")

        # Store results
        self.results['Ensemble'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_topics': train_ensemble,
            'test_topics': test_ensemble,
            'component_models': model_names
        }

        return train_ensemble

    def run_all(self, train_data, test_data):
        """
        Run the complete topic modeling study
        """
        print("=== AG-News Topic Modeling ===\n")

        # Load and preprocess data
        self.load_and_preprocess_data(train_data, test_data)

        # Run all implemented methods
        print("\n" + "=" * 50)
        self.run_bertopic()

        print("\n" + "=" * 50)
        self.run_top2vec()

        # doesn't work yet, maybe just leave out, mistiges octis
        print("\n" + "=" * 50)
        #self.implement_neural_topic_model()

        print("\n" + "=" * 50)
        self.run_transformer_topic_modeling()

        print("\n" + "=" * 50)
        self.implement_ensemble_methods()

        # Generate final comparison
        self.generate_final_comparison()

    def generate_final_comparison(self):
        """
        Generate comprehensive comparison of all methods
        """
        print("\n" + "=" * 60)
        print("FINAL PERFORMANCE COMPARISON")
        print("=" * 60)

        if not self.results:
            print("No results to compare.")
            return

        # Create comparison dataframe
        comparison_data = []

        for model_name, result in self.results.items():
            if 'test_metrics' in result:
                row = {'Model': model_name}
                row.update(result['test_metrics'])
                comparison_data.append(row)

        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison = df_comparison.round(4)

            print("\nTest Set Performance Comparison:")
            print(df_comparison.to_string(index=False))

            # Find best performing model for each metric
            print("\n" + "-" * 40)
            print("BEST PERFORMING MODELS:")
            print("-" * 40)

            metrics = ['ARI', 'NMI', 'Homogeneity', 'Completeness', 'V-Measure']
            for metric in metrics:
                if metric in df_comparison.columns:
                    best_idx = df_comparison[metric].idxmax()
                    best_model = df_comparison.loc[best_idx, 'Model']
                    best_score = df_comparison.loc[best_idx, metric]
                    print(f"{metric}: {best_model} ({best_score:.4f})")

            # Plot comparison
            self.plot_performance_comparison(df_comparison)

    def plot_performance_comparison(self, df_comparison):
        """
        Plot performance comparison across all models
        """
        if len(df_comparison) == 0:
            return

        metrics = ['ARI', 'NMI', 'Homogeneity', 'Completeness', 'V-Measure']
        available_metrics = [m for m in metrics if m in df_comparison.columns]

        if not available_metrics:
            return

        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 6))

        if len(available_metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            bars = ax.bar(df_comparison['Model'], df_comparison[metric])
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)

            # Color the best performing bar
            best_idx = df_comparison[metric].idxmax()
            bars[best_idx].set_color('red')

            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to run the comprehensive topic modeling study
    """
    print("AG-News Topic Modeling Study")
    print("=" * 50)

    # Initialize the study
    study = AGNewsTopicModeling()

    # Load data (replace with actual AG-News data loading)
    print("Loading AG-News dataset...")

    print("--- Loading and Preparing Data ---")

    path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
    train_df, test_df = data_processing.read_ag_news_split(path)
    subsample = True
    if subsample:
        print("SUBSAMPLING")
        train_df_subset, _ = train_test_split(
            train_df,
            train_size=12000,
            stratify=train_df['labels'],
            random_state=42
        )

        train_df, test_df = train_test_split(
            train_df_subset,
            train_size=10000,
            stratify=train_df_subset['labels'],
            random_state=42
        )

    print(train_df.shape)
    processing_cols = ['Title', 'Description']

    # clean text columns - try to remove html stuff
    train_df["Title"] = train_df["Title"].apply(lambda t: data_processing.clean_given_text(t) if pd.notnull(t) else t)
    train_df["Description"] = train_df["Description"].apply(lambda d: data_processing.clean_given_text(d) if pd.notnull(d) else d)
    test_df["Title"] = test_df["Title"].apply(lambda t: data_processing.clean_given_text(t) if pd.notnull(t) else t)
    test_df["Description"] = test_df["Description"].apply(lambda d: data_processing.clean_given_text(d) if pd.notnull(d) else d)

    # more removal of unnecessary noise
    data_processing.remove_quot_occurences(train_df, processing_cols)
    data_processing.remove_quot_occurences(test_df, processing_cols)
    data_processing.replace_numeric_entities(train_df, processing_cols)
    data_processing.replace_numeric_entities(test_df, processing_cols)
    data_processing.remove_character_references(train_df, processing_cols)
    data_processing.remove_character_references(test_df, processing_cols)

    train_df['Combined'] = train_df['Title'] + ' ' + train_df['Description']
    test_df['Combined'] = test_df['Title'] + ' ' + test_df['Description']

    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Label distribution in training: {train_df['labels'].value_counts().to_dict()}")

    # Run all
    study.run_all(train_df, test_df)

    print("\n" + "=" * 60)
    print("RUN ALL COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return study

if __name__ == "__main__":
    # Run the main study
    main()