"""
1_approach Package - LDA Implementation
"""

import sys
import os

# Add parent directory to access data_pipeline
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from parent data_pipeline
from data_pipeline.data_processing import (
    load_data_from_kagglehub, read_ag_news_split, initialize_nltk,
    get_stopwords_stemmer_lemmatizer, preprocess_text_stemming,
    remove_character_references, remove_quot_occurences,
    initialize_vektorizer_tfidf, apply_tfidf_single_train_test,
    AG_LABELS
)

from data_pipeline.visualizations import (
    visualize_word_distribution_for, compare_wordclouds_per_class
)

# Import local LDA implementation
from implementation_approach_1 import (
    LDATopicModeler, LDAComparison, LDAEvaluator, run_complete_lda_pipeline
)

__version__ = "1.0.0"