import os
import umap
import hdbscan

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import kagglehub
import pandas as pd
import numpy as np
import nltk
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
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
from wordcloud import WordCloud
from matplotlib.gridspec import GridSpec
# Example usage:
# visualize_top2vec_reduction(reduction_result, num_topics=4, num_words=8)


from bs4 import BeautifulSoup
from io import StringIO
import html

import urllib.parse
import pandas as pd
import numpy as np
import re

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

        # AG-News categories starting with 0
        self.category_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

    def _preprocess_text(self, texts):
        processed_texts = []
        numeric_entity_pattern = r'#(\d+);'
        # print("PREPROCESSING TEXTS...")
        for i, text in enumerate(texts):
            # Replace numeric entities
            def replace_entity(match):
                return chr(int(match.group(1)))

            clean_text = re.sub(numeric_entity_pattern, replace_entity, text)
            # Remove common web-related tokens and first-level domains
            clean_text = re.sub(r'\b(http|www|href|aspx|com|org|net|edu|gov|info|biz)\b', '', clean_text,
                                flags=re.IGNORECASE)
            # Remove weekdays (full and abbreviations)
            clean_text = re.sub(
                r'\b(Monday|Mon|Tuesday|Tue|Tues|Wednesday|Wed|Thursday|Thu|Thurs|Friday|Fri|Saturday|Sat|Sunday|Sun)\b',
                '', clean_text, flags=re.IGNORECASE)
            # Remove extra whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            clean_text = str(clean_text).lower()

            # Remove special characters and numbers (if still there)
            clean_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                                ' URL ', clean_text)
            clean_text = re.sub(r'\S+@\S+', ' EMAIL ', clean_text)
            clean_text = clean_text.split()
            # remove single letter words. 2 and 3 letter words may stay, as eu, us etc are valid
            clean_text = [word for word in clean_text if
                          len(word) > 1 and word not in ['and', 'to', 'in', 'the', 'of', 'for', 'on', 'as', 'is', 'ap',
                                                         'by', 'has', 'reuters', 'that', 'an', 'was', 'at', 'with',
                                                         'its', 'be', 'from', 'it', 'his', 'will', 'are', 'he', 'have',
                                                         'this', 'but']]
            clean_text = " ".join(clean_text).strip()
            processed_texts.append(clean_text)
        return processed_texts

    def load_and_preprocess_data(self, train_data, test_data, text_column='Description', label_column='labels'):
        """
        Load and preprocess AG-News data with given preprocessing pipeline
        """
        print("Loading and preprocessing AG-News data...")
        # Extract texts and labels

        print("train df ", train_data.shape)
        print("test df ", test_data.shape)

        # make labels fit to standard conventions starting with 0 (for bertopic etc easier)
        train_data[label_column] = train_data[label_column].apply(lambda x: x - 1)
        test_data[label_column] = test_data[label_column].apply(lambda x: x - 1)

        print("LABELS are:")
        print(set(train_data[label_column].tolist()))

        texts = train_data[text_column].tolist()
        # print("TEXTS from train_df", texts)
        self.labels = train_data[label_column].tolist()
        processed_texts = self._preprocess_text(texts)

        test_texts = test_data[text_column].tolist()
        self.test_labels = test_data[label_column].tolist()
        processed_test_texts = self._preprocess_text(test_texts)

        self.processed_texts = processed_texts
        self.test_texts = processed_test_texts
        # print("Train")
        # print(processed_texts)
        # print("Test")
        # print(processed_test_texts)

        # Find optimal k
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Generate embeddings for optimal k finding
        print("Generating embeddings for optimal k calculation...")
        embeddings = embedding_model.encode(self.processed_texts)
        self.elbow_k, silhouette_k, _, _ = self.find_optimal_clusters_elbow(embeddings)
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
        ax2.set_title(f'Silhouette Score vs Number of Clusters - {self.column}')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"elbow_vs_silhouette_{self.column}.png", dpi=300, bbox_inches="tight", pad_inches=0.25,
                    facecolor="white", )
        plt.show()

        return elbow_k, best_silhouette_k, sse, silhouette_scores

    def evaluate_topic_model(self, predicted_topics, true_labels, model_name):
        """
        Enhanced evaluation with topic-label alignment and additional metrics
        """
        predicted_topics = np.array(predicted_topics)
        true_labels = np.array(true_labels)

        # Remove any samples where preprocessing failed (empty predictions)
        valid_mask = (predicted_topics != -1) & (true_labels != -1)
        predicted_topics = predicted_topics[valid_mask]
        true_labels = true_labels[valid_mask]

        if len(predicted_topics) == 0:
            print(f"No valid predictions for {model_name}")
            return {}

        # Standard clustering metrics
        ari = adjusted_rand_score(true_labels, predicted_topics)
        nmi = normalized_mutual_info_score(true_labels, predicted_topics)
        homogeneity = homogeneity_score(true_labels, predicted_topics)
        completeness = completeness_score(true_labels, predicted_topics)
        v_measure = v_measure_score(true_labels, predicted_topics)

        # Hungarian algorithm for optimal topic-label alignment
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predicted_topics)

        # Find optimal assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-cm)  # Negative for maximization

        # Calculate accuracy with optimal alignment
        aligned_accuracy = cm[row_ind, col_ind].sum() / cm.sum()

        # Calculate per-class metrics with alignment
        aligned_predictions = np.zeros_like(predicted_topics)
        topic_to_label_map = {}

        # print("PREDICTED TOPICS: ", set(predicted_topics))

        for true_label, pred_topic in zip(row_ind, col_ind):
            topic_to_label_map[pred_topic] = true_label
            aligned_predictions[predicted_topics == pred_topic] = true_label

        # Per-class precision, recall, F1
        class_report = classification_report(true_labels, aligned_predictions,
                                             target_names=[self.category_names[i] for i in range(4)],
                                             output_dict=True, zero_division=0)

        metrics = {
            'ARI': ari,
            'NMI': nmi,
            'Homogeneity': homogeneity,
            'Completeness': completeness,
            'V-Measure': v_measure,
            'Aligned_Accuracy': aligned_accuracy,
            'Macro_F1': class_report['macro avg']['f1-score'],
            'Weighted_F1': class_report['weighted avg']['f1-score']
        }

        print(f"\n{model_name} Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Show topic-label alignment
        # print("TOPIC-LABEL-ALIGNMENT-MAP")
        # print(topic_to_label_map)
        print(f"\nOptimal Topic-Label Alignment:")
        for pred_topic, true_label in topic_to_label_map.items():
            print(f"Topic {pred_topic} -> {self.category_names[true_label]} "
                  f"({cm[true_label, pred_topic]} samples)")

        # Create and save confusion matrix plot
        self.plot_confusion_matrix(cm, row_ind, col_ind, model_name)

        return metrics, aligned_predictions

    def plot_confusion_matrix(self, cm, row_ind, col_ind, model_name):
        """
        Plot confusion matrix with optimal alignment highlighted
        """
        # Create figure with adjusted layout
        fig = plt.figure(figsize=(16, 6))

        # Create aligned confusion matrix for visualization
        aligned_cm = np.zeros_like(cm)
        for i, j in zip(row_ind, col_ind):
            aligned_cm[i, j] = cm[i, j]

        # Plot original confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Topic {i}' for i in range(cm.shape[1])],
                    yticklabels=[self.category_names[i] for i in range(cm.shape[0])])
        plt.title(f'{model_name} - Original CM {self.column} \n', fontsize=12, pad=20)  # Added padding
        plt.ylabel('True Labels', fontsize=10)
        plt.xlabel('Predicted Topics', fontsize=10)

        # Plot with optimal alignment highlighted
        plt.subplot(1, 2, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Topic {i}' for i in range(cm.shape[1])],
                    yticklabels=[self.category_names[i] for i in range(cm.shape[0])])

        # Highlight optimal alignment
        for i, j in zip(row_ind, col_ind):
            plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                              edgecolor='red', lw=3))

        plt.title(f'{model_name} - Optimal Alignment - {self.column}\n', fontsize=12, pad=20)  # Added padding
        plt.ylabel('True Labels', fontsize=10)
        plt.xlabel('Predicted Topics', fontsize=10)

        # Adjust layout with more padding
        plt.tight_layout(pad=3.0, w_pad=5.0, h_pad=2.0)

        # Save with proper bounding box
        plt.savefig(f"{model_name.replace(' ', '_')}_confusion_matrix_{self.column}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_topic_sizes_bert(self, model, model_name, num_topics=None):
        """
        Plot distribution of topic sizes for BERTopic including outliers

        Args:
            model: Trained BERTopic model
            model_name: Name of the model for title/saving
            num_topics: Number of topics to display (None for all)
        """
        # Get topic info
        topic_info = model.get_topic_info()
        topic_sizes = topic_info.set_index('Topic')['Count'].to_dict()

        # Prepare data
        sizes = []
        labels = []
        colors = []

        # add outliers
        if -1 in topic_sizes:
            sizes.append(topic_sizes[-1])
            labels.append("Outliers (-1)")
            colors.append('#cccccc')  # Gray for outliers

        # Add regular topics
        regular_topics = [(topic, size) for topic, size in topic_sizes.items() if topic != -1]

        # Sort by size (descending)
        regular_topics.sort(key=lambda x: x[1], reverse=True)

        # Limit number of topics if specified
        if num_topics is not None:
            regular_topics = regular_topics[:num_topics]

        # Add to plot data
        for topic, size in regular_topics:
            sizes.append(size)
            labels.append(f"Topic {topic}")

        # Create color gradient for regular topics
        num_regular = len(regular_topics)
        if num_regular > 0:
            regular_colors = plt.cm.tab20(np.linspace(0, 1, num_regular))
            colors.extend(regular_colors)

        # Plot
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(sizes))

        bars = plt.barh(y_pos, sizes, color=colors)
        plt.yticks(y_pos, labels)
        plt.gca().invert_yaxis()

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + max(sizes) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{int(width)}',
                     va='center', ha='left')

        plt.xlabel('Number of Documents')
        plt.title(f'{model_name} - Topic Distribution {self.column} \n(Total documents: {sum(sizes)})')
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        # Adjust margins and save
        plt.tight_layout()
        plt.savefig(f"{model_name.replace(' ', '_')}_topic_distribution_{self.column}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_topic_words_bert(self, model, model_name, num_topics=None, num_words=10):
        """
        Plot top words for each topic in a table format including outliers

        Args:
            model: Trained BERTopic model
            model_name: Name for title/saving
            num_topics: Number of topics to display
            num_words: Number of words per topic to show
        """
        # Get topic info
        topic_info = model.get_topic_info()

        # Prepare topics to show
        topics_to_show = [t for t in topic_info['Topic'] if t != -1][:num_topics]
        if -1 in topic_info['Topic'].values:
            topics_to_show = [-1] + topics_to_show  # Add outliers first

        fig, ax = plt.subplots(figsize=(10, max(6, len(topics_to_show) * 0.5)))

        # Prepare colors (gray for outliers)
        colors = []
        if -1 in topics_to_show:
            colors.append('#cccccc')  # Gray for outliers
        colors.extend(plt.cm.tab20(np.linspace(0, 1, len(topics_to_show) - len(colors))))

        cell_text = []
        for topic in topics_to_show:
            words_scores = model.get_topic(topic)
            words = [word for word, score in words_scores[:num_words]]
            scores = [score for word, score in words_scores[:num_words]]
            cell_text.append([f"{word} ({score:.2f})" for word, score in zip(words, scores)])

        # Create table
        table = ax.table(
            cellText=cell_text,
            rowLabels=[f"Outliers (-1)" if topic == -1 else f"Topic {topic}"
                       for topic in topics_to_show],
            rowColours=colors,
            colLabels=[f"Rank {i + 1}" for i in range(num_words)],
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.axis('off')
        plt.title(f'{model_name} Top {num_words} Words per Topic - {self.column}', pad=20)
        plt.tight_layout()
        plt.savefig(f"{model_name.replace(' ', '_')}_words_per_topic_{self.column}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_wordclouds_bert(self, model, model_name, num_topics=None, words_per_cloud=20):
        """
        Generate word clouds for each topic including outliers

        Args:
            model: Trained BERTopic model
            model_name: Name for title/saving
            num_topics: Number of topics to visualize
            words_per_cloud: Number of words to include in each word cloud
        """
        # Get topic info
        topic_info = model.get_topic_info()

        # Prepare topics to show
        topics_to_show = [t for t in topic_info['Topic'] if t != -1][:num_topics]
        if -1 in topic_info['Topic'].values:
            topics_to_show = [-1] + topics_to_show  # Add outliers first

        rows = (len(topics_to_show) + 1) // 2
        plt.figure(figsize=(16, 4 * rows))
        plt.suptitle(f"{model_name} Topic Word Clouds", y=1.02)

        for i, topic in enumerate(topics_to_show):
            plt.subplot(rows, 2, i + 1)
            words_scores = model.get_topic(topic)
            word_weights = {word: score for word, score in words_scores[:words_per_cloud]}

            # Custom color for outliers
            colormap = 'Greys' if topic == -1 else 'tab20'

            wc = WordCloud(
                width=600,
                height=300,
                background_color='white',
                colormap=colormap
            )
            wc.generate_from_frequencies(word_weights)
            plt.imshow(wc, interpolation='bilinear')
            plt.title(f"Outliers (-1)" if topic == -1 else f"Topic {topic}", fontsize=12)
            plt.axis('off')

        plt.tight_layout(pad=2.0)
        plt.savefig(f"{model_name.replace(' ', '_')}_topic_wordclouds_{self.column}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def run_bertopic(self, model_name):
        """
        BERTopic with AG-News embeddings
        """
        if not BERTOPIC_AVAILABLE:
            print("BERTopic not available. Skipping...")
            return None, None

        print(f"\n=== Running BERTopic {model_name} ===")

        # Use sentence transformer for embeddings
        embedding_model = SentenceTransformer(model_name)

        umap_model = umap.UMAP(
            n_neighbors=10,
            n_components=5,
            metric='cosine',
            random_state=42
        )

        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            metric='euclidean',
            cluster_selection_method="eom",
            prediction_data=True
        )

        # Initialize BERTopic with optimal number of topics
        topic_model = BERTopic(
            nr_topics=self.elbow_k + 1,  # + 1 for outlier class
            min_topic_size=2,
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            verbose=True
        )

        # Fit the model
        print(f"Fitting BERTopic {model_name} model...")
        topics, probs = topic_model.fit_transform(self.processed_texts)

        # Test on test set
        test_topics, test_probs = topic_model.transform(self.test_texts)

        # Get predicted topic counts
        topic_counts = Counter(topics)
        print(f"Predicted topic counts (train): {topic_counts}")

        # Get predicted topic counts
        topic_counts = Counter(test_topics)
        print(f"Predicted topic counts (test): {test_topics}")

        # Handle outliers (-1 topics) by assigning to closest topic
        if -1 in topics:
            print(f"Found {sum(t == -1 for t in topics)} outlier documents in training")
            # Assign outliers to most probable topic
            for i, (topic, prob_dist) in enumerate(zip(topics, probs)):
                if topic == -1:
                    topics[i] = np.argmax(prob_dist)

        if -1 in test_topics:
            print(f"Found {sum(t == -1 for t in test_topics)} outlier documents in test")
            for i, (topic, prob_dist) in enumerate(zip(test_topics, test_probs)):
                if topic == -1:
                    test_topics[i] = np.argmax(prob_dist)

        # Evaluate performance
        train_metrics, _ = self.evaluate_topic_model(topics, self.labels, f"(Train)-{model_name}")
        test_metrics, _ = self.evaluate_topic_model(test_topics, self.test_labels, f"(Test)-{model_name}")

        self.plot_topic_sizes_bert(topic_model, model_name=model_name, num_topics=None)

        self.plot_topic_words_bert(topic_model, model_name=model_name, num_topics=None, num_words=5)
        self.plot_wordclouds_bert(topic_model, model_name=model_name, num_topics=None, words_per_cloud=20)

        self.results[f'BERT_{model_name}'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_topics': topics,
            'test_topics': test_topics,
            'model': topic_model
        }

        # UMAP-Reduction
        embeddings = embedding_model.encode(self.processed_texts)
        reduced_embeddings = umap_model.fit_transform(embeddings)
        reduced_embeddings_embeddings_test = umap_model.transform(embedding_model.encode(self.test_texts))

        _, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=self.labels, cmap='tab10', s=10)
        axes[0].set_title(f"UMAP Projection (Ground Truth) - Train - {self.column}")
        axes[1].scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=topics, cmap='tab10', s=10)
        axes[1].set_title(f"UMAP Projection ({model_name}) - Train - {self.column}")
        plt.savefig(f"{model_name}-UMAP-Projection-Train_{self.column}.png")

        plt.show()

        _, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].scatter(reduced_embeddings_embeddings_test[:, 0], reduced_embeddings_embeddings_test[:, 1],
                        c=self.test_labels, cmap='tab10', s=10)
        axes[0].set_title(f"UMAP Projection (Ground Truth) - Test - {self.column}")
        axes[1].scatter(reduced_embeddings_embeddings_test[:, 0], reduced_embeddings_embeddings_test[:, 1],
                        c=test_topics, cmap='tab10', s=10)
        axes[1].set_title(f"UMAP Projection ({model_name}) - Test - {self.column}")
        plt.savefig(f"{model_name}-UMAP-Projection-Test-{self.column}.png")
        plt.show()

        # Display topic information
        print("\nTop words per topic:")
        topic_info = topic_model.get_topic_info()
        print(topic_info.head(10))

        # Show representative documents for each topic
        for topic_id in range(self.elbow_k):
            print(f"\nTopic {topic_id} - Representative documents:")
            topic_docs = [doc for i, doc in enumerate(self.processed_texts) if topics[i] == topic_id]
            if topic_docs:
                print(f"  Example: {topic_docs[0][:200]}...")

        # Quick evaluation
        test_nmi = normalized_mutual_info_score(self.test_labels, test_topics)
        print(f"{model_name} Test NMI: {test_nmi:.4f}")

        return test_nmi, topic_model, topics, test_topics

    def run_top2vec(self, speed="learn"):
        """
        Top2Vec Implementation
        """
        if not TOP2VEC_AVAILABLE:
            print("Top2Vec not available. Skipping...")
            return None, None

        print("\n=== Running Top2Vec for Exploration ===")
        print(f"Training Top2Vec on {len(self.processed_texts)} documents...")
        umap_args = {'n_neighbors': 10,
                     'n_components': 5,
                     'metric': 'cosine',
                     "random_state": 42}
        hdbscan_args = {'min_cluster_size': 10,
                        'min_samples': 5,
                        'metric': 'euclidean',
                        'cluster_selection_method': 'eom'}
        model = Top2Vec(documents=self.processed_texts,
                        speed=speed,
                        workers=2,
                        embedding_model='universal-sentence-encoder',
                        umap_args=umap_args,
                        hdbscan_args=hdbscan_args,
                        min_count=2,
                        embedding_batch_size=1000,
                        split_documents=False,
                        verbose=True
                        )

        original_topics = model.get_num_topics()
        print(f"Top2Vec found {original_topics} topics")

        if original_topics > self.elbow_k:
            print("Before reduction:", model.get_num_topics())
            result = model.hierarchical_topic_reduction(num_topics=self.elbow_k)
            print("After reduction:", len(result))
            print("Reduction result:", result)
            self.visualize_top2vec_reduction(model, num_topics=self.elbow_k, num_words=8)

        elif original_topics < self.elbow_k:
            print(f"Warning: Less than {self.elbow_k} topics found - data might not support  {self.elbow_k} clusters")

        train_metrics, _ = self.evaluate_topic_model(model.doc_top_reduced, self.labels, "Top2Vec (Train)")

        # Test set evaluation
        test_metrics = None
        if self.test_texts is not None:
            print("\nEvaluating on test set...")

            # Get topic assignments for test documents
            test_doc_topics, test_doc_scores = [], []

            for doc in self.test_texts:
                try:
                    _, _, topic_scores, topic_nums = model.query_topics(doc, num_topics=1)

                    if len(topic_nums) > 0:
                        test_doc_topics.append(topic_nums[0])
                        test_doc_scores.append(topic_scores[0])
                    else:
                        test_doc_topics.append(-1)  # No topic found
                        test_doc_scores.append(0)

                except Exception as e:
                    print(f"Error processing document: {str(e)[:100]}...")
                    test_doc_topics.append(-1)
                    test_doc_scores.append(0)

            if self.test_labels is not None:
                topic_mapping = {}
                for reduced_idx, sublist in enumerate(result):
                    for topic_num in sublist:
                        topic_mapping[topic_num] = reduced_idx

                test_doc_topics = [topic_mapping[topic_num] for topic_num in test_doc_topics]
                test_metrics, _ = self.evaluate_topic_model(test_doc_topics, self.test_labels, "Top2Vec (Test)")

                # Print assignment quality stats
                assigned_mask = np.array(test_doc_topics) != -1
                print(f"\nTest Set Assignment Quality:")
                print(f"Assigned documents: {sum(assigned_mask)}/{len(self.test_texts)}")
                print(f"Mean assignment score: {np.mean(np.array(test_doc_scores)[assigned_mask]):.2f}")
                print(f"Median assignment score: {np.median(np.array(test_doc_scores)[assigned_mask]):.2f}")

        umap_model = umap.UMAP(
            n_neighbors=10,
            n_components=5,
            metric='cosine',
            random_state=42
        )

        reduced_embeddings = umap_model.fit_transform(model.document_vectors)

        _, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=self.labels, cmap='tab10', s=10)
        axes[0].set_title(f"UMAP Projection (Ground Truth) - Train - {self.column}")
        axes[1].scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=model.doc_top_reduced, cmap='tab10', s=10)
        axes[1].set_title(f"UMAP Projection (Top2Vec)) - Train - {self.column}")
        plt.savefig(f"(Top2Vec)-UMAP-Projection-Train-{self.column}.png")
        plt.show()

        # Store results
        self.results['Top2Vec'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_topics': model.doc_top_reduced,
            'test_topics': test_doc_topics,
            'model': model,
            'test_assignments': list(
                zip(self.test_texts, test_doc_topics, test_doc_scores)) if self.test_texts else None
        }

        return model

    def plot_topic_sizes(self, model, num_topics=None):
        if num_topics is None:
            num_topics = len(model.topic_sizes_reduced)

        plt.figure(figsize=(8, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, num_topics))

        plt.barh(range(num_topics),
                 model.topic_sizes_reduced[:num_topics],
                 color=colors)

        plt.yticks(range(num_topics), [f"Topic {i}" for i in range(num_topics)])
        plt.gca().invert_yaxis()
        plt.xlabel('Number of Documents')
        plt.title(f'Top2Vec Topic Distribution - {self.column}')
        plt.tight_layout()
        plt.savefig(f"TOP2VEC-Topic-distribution-{self.column}.png")
        plt.show()

    def plot_topic_words(self, model, num_topics=None, num_words=10):
        if num_topics is None:
            num_topics = len(model.topic_words_reduced)

        fig, ax = plt.subplots(figsize=(10, max(6, num_topics * 0.5)))
        colors = plt.cm.tab20(np.linspace(0, 1, num_topics))

        cell_text = []
        for i in range(num_topics):
            words = model.topic_words_reduced[i][:num_words]
            scores = model.topic_word_scores_reduced[i][:num_words]
            cell_text.append([f"{word} ({score:.2f})" for word, score in zip(words, scores)])

        table = ax.table(cellText=cell_text,
                         rowLabels=[f"Topic {i}" for i in range(num_topics)],
                         rowColours=colors,
                         colLabels=[f"Rank {i + 1}" for i in range(num_words)],
                         loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.axis('off')
        plt.title(f'Top2Vec Top {num_words} Words per Topic - {self.column}', pad=20)
        plt.tight_layout()
        plt.savefig(f"TOP2VEC-words-per-topic-{self.column}.png")

        plt.show()

    def plot_wordclouds(self, model, num_topics=None, words_per_cloud=20):
        if num_topics is None:
            num_topics = len(model.topic_words_reduced)

        rows = (num_topics + 1) // 2
        plt.figure(figsize=(16, 4 * rows))

        for i in range(num_topics):
            plt.subplot(rows, 2, i + 1)
            word_weights = {word: score for word, score in
                            zip(model.topic_words_reduced[i][:words_per_cloud],
                                model.topic_word_scores_reduced[i][:words_per_cloud])}

            wc = WordCloud(width=600, height=300,
                           background_color='white', colormap='tab20')
            wc.generate_from_frequencies(word_weights)
            plt.imshow(wc, interpolation='bilinear')
            plt.title(f'Topic {i}', fontsize=12)
            plt.axis('off')

        plt.tight_layout(pad=2.0)
        plt.savefig(f"TOP2VEC-Topic-wordclouds-{self.column}.png")
        plt.show()

    def visualize_top2vec_reduction(self, model, num_topics=None, num_words=10):
        """
        Visualizes the reduced topics from a Top2Vec model after hierarchical reduction.

        Args:
            model: The Top2Vec model after hierarchical reduction
            num_topics: Number of topics to display (None for all)
            num_words: Number of top words to show in the table
        """
        if num_topics is None:
            num_topics = len(model.topic_words_reduced)

        self.plot_topic_sizes(model, self.elbow_k)
        self.plot_topic_words(model, self.elbow_k, num_words=5)
        self.plot_wordclouds(model, self.elbow_k)
        # self.evaluate_topic_model(model.doc_top_reduced, self.labels, "TOP2VEC")

    def run_transformer_topic_modeling(self):
        """
        Transformer-based topic modeling approaches with different models & BERTopic
        """
        print("\n=== Run Transformer-based Topic Modeling (Bertopic) ===")

        if not BERTOPIC_AVAILABLE:
            print("BERTopic not available for transformer-based approach. Skipping...")
            return None, None

        # Try different transformer models
        transformer_models = [
            'all-MiniLM-L12-v2',
            'all-mpnet-base-v2',
        ]

        best_model = None
        best_score = -1

        for model_name in transformer_models:
            print(f"Testing {model_name}...")
            test_nmi, topic_model, topics, test_topics = self.run_bertopic(model_name=model_name)
            # print(test_nmi)
            # topic_model.visualize_topics()

            if test_nmi > best_score:
                best_score = test_nmi
                best_model = topic_model
                best_train_topics = topics
                best_test_topics = test_topics
                best_model_name = model_name

        if best_model is not None:
            print(f"\nBest transformer model: {best_model_name}")
            train_metrics, _ = self.evaluate_topic_model(best_train_topics, self.labels, f"Transformer (Train)")
            test_metrics, _ = self.evaluate_topic_model(best_test_topics, self.test_labels, f"Transformer (Test)")

            self.results['Transformer'] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'model': best_model,
                'train_topics': best_train_topics,
                'test_topics': best_test_topics,
                'best_model_name': best_model_name
            }

        return

    def run_ensemble_methods(self):
        """
        Test ensemble methods combining different topic models
        """
        print("\n=== Running Ensemble Methods ===")

        if len(self.results) < 2:
            print("Need at least 2 models for ensemble. Skipping...")
            return None

        # Collect predictions from all available models
        train_predictions = []
        test_predictions = []
        model_names = []

        for model_name, result in self.results.items():
            if model_name not in ['Top2Vec', 'Transformer']:
                continue
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
        train_metrics, _ = self.evaluate_topic_model(train_ensemble, self.labels, "Ensemble (Train)")
        test_metrics, _ = self.evaluate_topic_model(test_ensemble, self.test_labels, "Ensemble (Test)")

        # Store results
        self.results['Ensemble'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_topics': train_ensemble,
            'test_topics': test_ensemble,
            'component_models': model_names
        }

        return

    def run_all(self):
        """
        Run the complete topic modeling suite
        """
        print("\n" + "=" * 50)
        self.run_top2vec()

        print("\n" + "=" * 50)
        self.run_transformer_topic_modeling()

        print("\n" + "=" * 50)
        self.run_ensemble_methods()

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
            if model_name == 'Transformer':
                continue
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

            metrics = ['ARI', 'NMI', 'Homogeneity', 'Completeness', 'V-Measure', 'Aligned_Accuracy']
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

        metrics = ['ARI', 'NMI', 'Homogeneity', 'Completeness', 'V-Measure', 'Aligned_Accuracy']
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
        plt.savefig(f"general_comparison_{self.column}.png")
        plt.show()

    def load_data(self, column='Combined'):

        print("Loading AG-News dataset...")

        print("--- Loading and Preparing Data ---")

        path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
        train_df, test_df = data_processing.read_ag_news_split(path)
        subsample = True
        if subsample:
            print("SUBSAMPLING")
            train_df, _ = train_test_split(
                train_df,
                train_size=20000,
                stratify=train_df['labels'],
                random_state=42
            )

            _, test_df = train_test_split(
                test_df,
                train_size=3600,
                stratify=test_df['labels'],
                random_state=42
            )

        print(train_df.shape)
        processing_cols = ['Title', 'Description']

        # clean text columns - try to remove html stuff
        train_df["Title"] = train_df["Title"].apply(
            lambda t: data_processing.clean_given_text(t) if pd.notnull(t) else t)
        train_df["Description"] = train_df["Description"].apply(
            lambda d: data_processing.clean_given_text(d) if pd.notnull(d) else d)
        test_df["Title"] = test_df["Title"].apply(lambda t: data_processing.clean_given_text(t) if pd.notnull(t) else t)
        test_df["Description"] = test_df["Description"].apply(
            lambda d: data_processing.clean_given_text(d) if pd.notnull(d) else d)

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
        print("training data cols: ", train_df.columns)
        print(f"Test data shape: {test_df.shape}")
        print("testing data cols: ", test_df.columns)
        print(f"Label distribution in training: {train_df['labels'].value_counts().to_dict()}")

        self.column = column
        self.load_and_preprocess_data(train_df, test_df, text_column=column)

        return train_df, test_df


def main():
    """
    Main function to run the comprehensive topic modeling study
    """
    print("AG-News Topic Modeling")
    print("=" * 50)

    tm = AGNewsTopicModeling()
    tm.load_data()

    # Run all
    tm.run_all()

    print("\n" + "=" * 60)
    print("RUN ALL COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return tm


if __name__ == "__main__":
    main()