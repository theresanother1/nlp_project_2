"""
Visualization Functions for Topic Model Evaluation
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict

def plot_final_results(df: pd.DataFrame):
    """Plot a bar chart comparison of all models across all key metrics."""
    if df.empty:
        print("No results to plot.")
        return
    
    metrics = [col for col in df.columns if col != 'Model']
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5 * n_metrics))
    if n_metrics == 1:
        axes = [axes]
        
    for i, metric in enumerate(metrics):
        sns.barplot(x='Model', y=metric, data=df, ax=axes[i])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig('final_results.png')
    plt.close()

def plot_topic_words(model_topics: Dict[int, List[str]], model_name: str):
    """Visualize the top words for each topic."""
    if not model_topics:
        return
        
    n_topics = len(model_topics)
    fig, axes = plt.subplots(n_topics, 1, figsize=(8, 2 * n_topics))
    if n_topics == 1:
        axes = [axes]
        
    for i, (topic_id, words) in enumerate(model_topics.items()):
        y_pos = np.arange(len(words))
        axes[i].barh(y_pos, np.arange(len(words), 0, -1), align='center')
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(words)
        axes[i].invert_yaxis()
        axes[i].set_xlabel('Importance')
        axes[i].set_title(f'{model_name} - Topic {topic_id}')
        
    plt.tight_layout()
    plt.savefig(f'{model_name}_topics.png')
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, model_name: str, class_names: List[str]):
    """A standardized function for plotting confusion matrices."""
    if cm.size == 0:
        return
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()
