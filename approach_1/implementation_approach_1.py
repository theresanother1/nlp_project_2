"""
LDA Topic Modeling Implementation - Robust Version for CML
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
# Import local LDA implementation
#from implementation_approach_1 import run_complete_lda_pipeline


import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel

# Import AG_LABELS
import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from data_pipeline.data_processing import AG_LABELS
except ImportError:
    AG_LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Science/Tech"}


class LDATopicModeler:
    """LDA wrapper for scikit-learn and Gensim"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.sklearn_model = None
        self.gensim_model = None
        self.dictionary = None
        
    def train_sklearn_lda(self, X_train: Any, n_topics: int = 4, max_iter: int = 100, **kwargs):
        """Train LDA with scikit-learn"""
        try:
            self.sklearn_model = LatentDirichletAllocation(
                n_components=n_topics, random_state=self.random_state,
                max_iter=max_iter, verbose=0, **kwargs
            )
            self.sklearn_model.fit(X_train)
            return self.sklearn_model.perplexity(X_train)
        except Exception as e:
            print(f"Sklearn LDA training failed: {e}")
            return np.nan
    
    def train_gensim_lda(self, corpus: List, dictionary: corpora.Dictionary, 
                        n_topics: int = 4, passes: int = 10, **kwargs):
        """Train LDA with Gensim"""
        try:
            self.dictionary = dictionary
            self.gensim_model = models.LdaModel(
                corpus=corpus, id2word=dictionary, num_topics=n_topics,
                random_state=self.random_state, passes=passes,
                per_word_topics=True, **kwargs
            )
            return np.exp(-self.gensim_model.log_perplexity(corpus))
        except Exception as e:
            print(f"Gensim LDA training failed: {e}")
            return np.nan
    
    def get_sklearn_topics(self, feature_names: List[str], n_words: int = 10):
        """Extract topics from scikit-learn model"""
        if self.sklearn_model is None:
            return {}
        topics = {}
        for topic_idx, topic in enumerate(self.sklearn_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            topics[topic_idx] = [feature_names[i] for i in top_words_idx]
        return topics
    
    def get_gensim_topics(self, n_words: int = 10):
        """Extract topics from Gensim model"""
        if self.gensim_model is None:
            return {}
        topics = {}
        for idx, topic in self.gensim_model.print_topics(num_topics=-1, num_words=n_words):
            words = [word.split('"')[1] for word in topic.split(' + ')]
            topics[idx] = words
        return topics
    
    def calculate_coherence(self, texts: List[List[str]]):
        """Calculate coherence score"""
        try:
            if self.gensim_model is None or self.dictionary is None:
                return np.nan
            coherence_model = CoherenceModel(
                model=self.gensim_model, texts=texts, 
                dictionary=self.dictionary, coherence='c_v'
            )
            return coherence_model.get_coherence()
        except Exception as e:
            print(f"Coherence calculation failed: {e}")
            return np.nan


class LDAComparison:
    """Compare different LDA configurations"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def compare_topic_numbers(self, X_train, X_test, train_corpus, test_corpus,
                            dictionary, texts, topic_range=[4, 6, 8], max_samples=5000):
        """Compare different numbers of topics with robust error handling"""
        print(f"Comparing topics: {topic_range} with max {max_samples} samples")
        results = []
        
        # Limit data for performance
        X_train_small = X_train[:max_samples]
        X_test_small = X_test[:1000]
        train_corpus_small = train_corpus[:max_samples]
        test_corpus_small = test_corpus[:1000]
        texts_small = texts[:500]
        
        for i, n_topics in enumerate(topic_range):
            print(f"Testing {n_topics} topics ({i+1}/{len(topic_range)})...")
            result = {'n_topics': n_topics}
            
            modeler = LDATopicModeler(self.random_state)
            
            # Scikit-learn LDA
            print(f"  Training sklearn LDA...")
            sk_perplexity = modeler.train_sklearn_lda(
                X_train_small, n_topics=n_topics, max_iter=20, learning_method='online'
            )
            
            if not np.isnan(sk_perplexity) and modeler.sklearn_model is not None:
                try:
                    result['sklearn_perplexity'] = modeler.sklearn_model.perplexity(X_test_small)
                    print(f"    Sklearn success: {result['sklearn_perplexity']:.1f}")
                except Exception as e:
                    print(f"    Sklearn test failed: {e}")
                    result['sklearn_perplexity'] = np.nan
            else:
                result['sklearn_perplexity'] = np.nan
                print(f"    Sklearn training failed")
            
            # Gensim LDA
            print(f"  Training gensim LDA...")
            gs_perplexity = modeler.train_gensim_lda(
                train_corpus_small, dictionary, n_topics=n_topics, passes=3
            )
            
            if not np.isnan(gs_perplexity) and modeler.gensim_model is not None:
                try:
                    result['gensim_perplexity'] = np.exp(-modeler.gensim_model.log_perplexity(test_corpus_small))
                    result['gensim_coherence'] = modeler.calculate_coherence(texts_small)
                    print(f"    Gensim success: perp={result['gensim_perplexity']:.1f}, coh={result['gensim_coherence']:.3f}")
                except Exception as e:
                    print(f"    Gensim test failed: {e}")
                    result['gensim_perplexity'] = np.nan
                    result['gensim_coherence'] = np.nan
            else:
                result['gensim_perplexity'] = np.nan
                result['gensim_coherence'] = np.nan
                print(f"    Gensim training failed")
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        print(f"\nComparison completed. Results:")
        print(results_df)
        return results_df
    
    def hyperparameter_tuning(self, X_train, cv=3):
        """Hyperparameter tuning for scikit-learn LDA"""
        print("Hyperparameter tuning...")
        
        param_grid = {
            'n_components': [4, 6],  # Reduced for stability
            'doc_topic_prior': [0.1, 1.0],
            'topic_word_prior': [0.1, 1.0],
            'learning_method': ['online']  # Only online for speed
        }
        
        def neg_perplexity_score(estimator, X, y=None):
            try:
                return -estimator.perplexity(X)
            except:
                return -1000  # Very bad score if fails
        
        lda_grid = LatentDirichletAllocation(random_state=42, max_iter=20, verbose=0)
        grid_search = GridSearchCV(
            estimator=lda_grid, param_grid=param_grid,
            scoring=neg_perplexity_score, cv=cv, verbose=0, n_jobs=1  # Single job for stability
        )
        
        try:
            X_small = X_train[:2000]  # Even smaller for stability
            grid_search.fit(X_small)
            
            return {
                'best_params': grid_search.best_params_,
                'best_perplexity': -grid_search.best_score_
            }
        except Exception as e:
            print(f"Hyperparameter tuning failed: {e}")
            return {
                'best_params': {'n_components': 4, 'doc_topic_prior': 1.0, 'topic_word_prior': 1.0, 'learning_method': 'online'},
                'best_perplexity': np.nan
            }


class LDAEvaluator:
    """Evaluate LDA models against AG-News categories"""
    
    def evaluate_against_labels(self, model: LDATopicModeler, X_test, y_true):
        """Evaluate sklearn LDA model against AG-News labels"""
        try:
            if model.sklearn_model is None:
                print("No sklearn model available for evaluation")
                return {'confusion_matrix': np.zeros((4,4)), 'optimal_accuracy': 0.0}
            
            doc_topic_probs = model.sklearn_model.transform(X_test)
            dominant_topics = np.argmax(doc_topic_probs, axis=1) + 1
            
            # Confusion matrix and optimal accuracy
            cm = confusion_matrix(y_true, dominant_topics, labels=[1, 2, 3, 4])
            cost_matrix = -cm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            optimal_accuracy = cm[row_indices, col_indices].sum() / cm.sum()
            
            return {
                'confusion_matrix': cm,
                'optimal_accuracy': optimal_accuracy
            }
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {'confusion_matrix': np.zeros((4,4)), 'optimal_accuracy': 0.0}
    
    def visualize_confusion_matrix(self, cm, title="AG-News vs LDA Topics"):
        """Visualize confusion matrix"""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d',
                        xticklabels=[f'Topic {i+1}' for i in range(cm.shape[1])],
                        yticklabels=list(AG_LABELS.values()), cmap='Blues')
            plt.title(title)
            plt.xlabel('Predicted LDA Topic')
            plt.ylabel('True AG-News Category')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Visualization failed: {e}")


def run_complete_lda_pipeline(X_train, X_test, train_corpus, test_corpus,
                             dictionary, texts, feature_names, y_test,
                             config=None):
    """
    Complete LDA pipeline with robust error handling
    """
    if config is None:
        config = {
            'topic_range': [4, 6],
            'max_samples': 3000,  # Conservative for CML
            'hyperparameter_tuning': False,  # Disabled by default for stability
            'evaluate_against_labels': True
        }
    
    print("Starting robust LDA pipeline...")
    print(f"Config: {config}")
    
    results = {}
    
    # 1. Compare topic numbers
    print("\n1. Comparing topic numbers...")
    comparator = LDAComparison()
    comparison_df = comparator.compare_topic_numbers(
        X_train, X_test, train_corpus, test_corpus, 
        dictionary, texts, config['topic_range'], config['max_samples']
    )
    results['comparison'] = comparison_df
    
    # 2. Hyperparameter tuning (optional)
    if config['hyperparameter_tuning']:
        print("\n2. Hyperparameter tuning...")
        tuning_results = comparator.hyperparameter_tuning(X_train)
        results['hyperparameter_tuning'] = tuning_results
    
    # 3. Train best model (with fallback)
    print("\n3. Selecting best model...")
    
    # Check if we have valid coherence scores
    valid_coherence = comparison_df['gensim_coherence'].dropna()
    
    if len(valid_coherence) > 0:
        best_n_topics = comparison_df.loc[comparison_df['gensim_coherence'].idxmax(), 'n_topics']
        print(f"Selected {best_n_topics} topics based on coherence")
    else:
        # Fallback to default
        best_n_topics = 4  # Default to AG-News categories
        print(f"No valid coherence scores found. Using fallback: {best_n_topics} topics")
    
    print(f"Training final model with {best_n_topics} topics...")
    
    best_modeler = LDATopicModeler()
    best_modeler.train_sklearn_lda(X_train, n_topics=best_n_topics, max_iter=50)
    best_modeler.train_gensim_lda(train_corpus, dictionary, n_topics=best_n_topics, passes=10)
    
    results['best_model'] = best_modeler
    results['best_n_topics'] = best_n_topics
    
    # 4. Extract topics
    print("\n4. Extracting topics...")
    sklearn_topics = best_modeler.get_sklearn_topics(feature_names, n_words=8)
    gensim_topics = best_modeler.get_gensim_topics(n_words=8)
    results['sklearn_topics'] = sklearn_topics
    results['gensim_topics'] = gensim_topics
    
    print(f"Found {len(sklearn_topics)} sklearn topics, {len(gensim_topics)} gensim topics")
    
    # 5. Evaluation
    if config['evaluate_against_labels']:
        print("\n5. Evaluating against AG-News labels...")
        evaluator = LDAEvaluator()
        evaluation = evaluator.evaluate_against_labels(best_modeler, X_test, y_test)
        if evaluation['optimal_accuracy'] > 0:
            evaluator.visualize_confusion_matrix(evaluation['confusion_matrix'])
        results['evaluation'] = evaluation
    
    print("\nPipeline completed successfully!")
    return results
