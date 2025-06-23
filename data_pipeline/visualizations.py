import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from wordcloud import WordCloud
import warnings

from data_pipeline.data_processing import AG_LABELS

warnings.simplefilter(action='ignore', category=FutureWarning)

"""
TODO: Adapt to other datasets, if we use them
"""


def visualize_word_distribution_for(df: pd.DataFrame, label: str, imagefile):
    class_indices = sorted(df[label].unique())  # Get unique Class Index values

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the 2x2 grid into a 1D array for easier iteration

    # Generate a word cloud for each Class Index
    for i, class_idx in enumerate(tqdm(class_indices, desc="Generating Word Clouds")):
        world = df[df[label] == class_idx]['Description']

        # Check if the filtered data is not empty
        if len(world) > 0:
            # Generate the word cloud
            wordcloud = WordCloud(
                min_font_size=3,  # Minimum font size
                max_words=2500,  # Maximum number of words
                width=800,  # Width of the word cloud
                height=800,  # Height of the word cloud
                background_color='white'  # Background color
            ).generate(" ".join(world.astype(str)))  # Combine text data into a single string

            # Display the word cloud in the corresponding subplot
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'Class Index {AG_LABELS[class_idx]}', fontsize=14)  # Add a title
            axes[i].axis('off')  # Hide axes
        else:
            print(f"No data found for Class Index {AG_LABELS[class_idx]}.")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(imagefile)
    plt.show()


def compare_wordclouds_per_class(train_df: pd.DataFrame, test_df: pd.DataFrame, labels: str):
    class_indices = sorted(train_df[labels].unique())  # Get unique Class Index values

    # Create a 4x2 grid of subplots (4 rows for classes, 2 columns for train/test)
    fig, axes = plt.subplots(4, 2, figsize=(15, 30))  # Adjust figsize as needed
    axes = axes.flatten()  # Flatten the grid into a 1D array for easier iteration

    # Generate word clouds for each Class Index in train and test sets
    for i, class_idx in enumerate(tqdm(class_indices, desc="Generating Word Clouds")):
        # Train set
        train_world = train_df[train_df[labels] == class_idx]['Description']
        if len(train_world) > 0:
            train_wordcloud = WordCloud(
                min_font_size=3,  # Minimum font size
                max_words=2500,  # Maximum number of words
                width=800,  # Width of the word cloud
                height=800,  # Height of the word cloud
                background_color='white'  # Background color
            ).generate(" ".join(train_world.astype(str)))  # Combine text data into a single string

            # Display the train word cloud in the corresponding subplot
            axes[2 * i].imshow(train_wordcloud, interpolation='bilinear')
            axes[2 * i].set_title(f'Train - label {AG_LABELS[class_idx]}', fontsize=14)  # Add a title
            axes[2 * i].axis('off')  # Hide axes
        else:
            print(f"No train data found for label {AG_LABELS[class_idx]}.")

        # Test set
        test_world = test_df[test_df[labels] == class_idx]['Description']
        if len(test_world) > 0:
            test_wordcloud = WordCloud(
                min_font_size=3,  # Minimum font size
                max_words=2500,  # Maximum number of words
                width=800,  # Width of the word cloud
                height=800,  # Height of the word cloud
                background_color='white'  # Background color
            ).generate(" ".join(test_world.astype(str)))  # Combine text data into a single string

            # Display the test word cloud in the corresponding subplot
            axes[2 * i + 1].imshow(test_wordcloud, interpolation='bilinear')
            axes[2 * i + 1].set_title(f'Test - label {AG_LABELS[class_idx]}', fontsize=14)  # Add a title
            axes[2 * i + 1].axis('off')  # Hide axes
        else:
            print(f"No test data found for label {AG_LABELS[class_idx]}.")

    plt.tight_layout()
    plt.show()

