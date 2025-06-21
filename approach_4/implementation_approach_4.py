"""
Main Evaluation Script for Topic Modeling
"""
import pandas as pd
from gensim.corpora import Dictionary

# Import from other approaches
from data_pipeline.data_processing import AG_LABELS
from approach_1.implementation_approach_1 import LDATopicModeler
from approach_3.implementation_approach_3 import AGNewsTopicModeling

# Import from this approach
from approach_4.metrics import calculate_coherence, calculate_topic_diversity, evaluate_clustering_metrics, benchmark_model
from approach_4.visualizations import plot_final_results

def run_evaluation():
    """
    Run the full evaluation pipeline.
    """
    # --- 1. Load and Preprocess Data ---
    print("--- 1. Loading and Preprocessing Data ---")
    train_df = pd.read_csv('data/train_cleaned.csv')
    test_df = pd.read_csv('data/test_cleaned.csv')
    
    train_df['Combined'] = train_df['Title'] + ' ' + train_df['Description']
    test_df['Combined'] = test_df['Title'] + ' ' + test_df['Description']
    
    train_texts = [text.split() for text in train_df['Combined']]
    dictionary = Dictionary(train_texts)
    
    results = []

    # --- 2. Evaluate Approach 1: LDA ---
    print("\n--- 2. Evaluating Approach 1: LDA ---")
    def train_lda():
        modeler = LDATopicModeler()
        # A simple TF-IDF vectorizer for LDA
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X_train = vectorizer.fit_transform(train_df['Combined'])
        modeler.train_sklearn_lda(X_train, n_topics=4)
        # Pass vectorizer and feature_names along with the modeler
        return modeler, vectorizer
    
    lda_benchmark = benchmark_model(train_lda)
    lda_modeler, lda_vectorizer = lda_benchmark['model_output']
    
    if lda_modeler.sklearn_model:
        feature_names = lda_vectorizer.get_feature_names_out()
        lda_topics_list = list(lda_modeler.get_sklearn_topics(feature_names).values())
        
        lda_coherence = calculate_coherence(lda_topics_list, train_texts, dictionary)
        lda_diversity = calculate_topic_diversity(lda_topics_list)
        
        X_test = lda_vectorizer.transform(test_df['Combined'])
        doc_topic_dist = lda_modeler.sklearn_model.transform(X_test)
        predicted_topics = doc_topic_dist.argmax(axis=1)
        
        # Ensure 'Class Index' is numeric and handle NaNs
        test_df['Class Index'] = pd.to_numeric(test_df['Class Index'], errors='coerce')
        test_df.dropna(subset=['Class Index'], inplace=True)
        
        # Align predicted topics with the cleaned test_df
        predicted_topics = predicted_topics[test_df.index]
        
        lda_clustering_metrics = evaluate_clustering_metrics(predicted_topics, test_df['Class Index'] - 1)
        
        results.append({
            'Model': 'LDA (sklearn)',
            'Coherence': lda_coherence,
            'Diversity': lda_diversity,
            **lda_clustering_metrics,
            'Runtime (s)': lda_benchmark['runtime'],
            'Memory (MB)': lda_benchmark['memory_usage_mb']
        })

    # --- 3. Evaluate Approach 3: BERTopic ---
    print("\n--- 3. Evaluating Approach 3: BERTopic ---")
    def train_bertopic():
        model = AGNewsTopicModeling()
        # Workaround for hardcoded 'labels' column in 3_approach
        train_df_bertopic = train_df.rename(columns={'Class Index': 'labels'})
        test_df_bertopic = test_df.rename(columns={'Class Index': 'labels'})
        model.load_and_preprocess_data(train_df_bertopic, test_df_bertopic)
        model.run_bertopic(n_topics=4)
        return model

    bertopic_benchmark = benchmark_model(train_bertopic)
    bertopic_model = bertopic_benchmark['model_output']
    
    if 'BERTopic' in bertopic_model.results:
        bertopic_results = bertopic_model.results['BERTopic']
        
        bertopic_topics = bertopic_model.results['BERTopic']['model'].get_topics()
        bertopic_topic_words = [list(dict(topic).keys()) for topic in bertopic_topics.values()]

        bertopic_coherence = calculate_coherence(bertopic_topic_words, train_texts, dictionary)
        bertopic_diversity = calculate_topic_diversity(bertopic_topic_words)
        
        results.append({
            'Model': 'BERTopic',
            'Coherence': bertopic_coherence,
            'Diversity': bertopic_diversity,
            **bertopic_results['test_metrics'],
            'Runtime (s)': bertopic_benchmark['runtime'],
            'Memory (MB)': bertopic_benchmark['memory_usage_mb']
        })

    # --- 4. Final Comparison ---
    print("\n--- 4. Final Comparison ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_markdown(index=False))
    
    plot_final_results(results_df)
    
    return results_df

if __name__ == '__main__':
    run_evaluation()
