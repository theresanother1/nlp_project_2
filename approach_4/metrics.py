"""
Centralized Metrics for Topic Model Evaluation
"""
import time
import psutil
import numpy as np
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from typing import List, Dict, Callable, Any

def calculate_coherence(topics: List[List[str]], texts: List[List[str]], dictionary: Dictionary) -> float:
    """Calculate c_v coherence for a list of topics."""
    if not topics or not texts:
        return np.nan
    try:
        coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
        return coherence_model.get_coherence()
    except Exception as e:
        print(f"Error calculating coherence: {e}")
        return np.nan

def calculate_topic_diversity(topics: List[List[str]]) -> float:
    """Calculate topic diversity (percentage of unique words)."""
    if not topics:
        return 0.0
    unique_words = set()
    total_words = 0
    for topic in topics:
        unique_words.update(topic)
        total_words += len(topic)
    if total_words == 0:
        return 0.0
    return len(unique_words) / total_words

def evaluate_clustering_metrics(predicted_topics: List[int], true_labels: List[int]) -> Dict[str, float]:
    """Evaluate clustering performance against ground truth labels."""
    if len(predicted_topics) != len(true_labels):
        raise ValueError("Predicted topics and true labels must have the same length.")
    
    return {
        'ARI': adjusted_rand_score(true_labels, predicted_topics),
        'NMI': normalized_mutual_info_score(true_labels, predicted_topics),
        'Homogeneity': homogeneity_score(true_labels, predicted_topics),
        'Completeness': completeness_score(true_labels, predicted_topics),
        'V-Measure': v_measure_score(true_labels, predicted_topics)
    }

def benchmark_model(model_func: Callable[..., Any], *args, **kwargs) -> Dict[str, Any]:
    """Benchmark a model's training time and memory usage."""
    process = psutil.Process()
    start_mem = process.memory_info().rss
    start_time = time.time()
    
    model_output = model_func(*args, **kwargs)
    
    end_time = time.time()
    end_mem = process.memory_info().rss
    
    return {
        'model_output': model_output,
        'runtime': end_time - start_time,
        'memory_usage_mb': (end_mem - start_mem) / (1024 * 1024)
    }
