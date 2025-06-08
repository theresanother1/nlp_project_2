from typing import List

import pandas as pd
import kagglehub
import html
import urllib.parse
import re
import numpy as np
from bs4 import BeautifulSoup
from io import StringIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import json
import os

AG_LABELS = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Science/Tech"
}


def load_data_from_kagglehub():
    # used like this, so that it doesn't matter where the code is executed, as long as internet connection exists
    path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
    path_2 = kagglehub.dataset_download("alfathterry/bbc-full-text-document-classification")
    path_3 = kagglehub.dataset_download("rmisra/news-category-dataset")
    print("Path to ag-news files:", path)
    print("Path to bbc-news files:", path_2)
    print("Path to news-category files:", path_3)
    return path, path_2, path_3

def convert_json_to_csv(news_category_path: str):
    news_category_file = "News_Category_Dataset_v3.json"
    news_category_csv = 'News_Category_Dataset_v3.csv'
    with open(os.path.join(news_category_path, news_category_file), 'r') as file:
        data = [json.loads(line) for line in file]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    df.to_csv(news_category_csv, index=False)
    return news_category_csv


def read_ag_news_combined(path: str):
    # The first column is Class Index, the second column is Title and the third column is Description.
    # The class ids are numbered 1-4 where
    # 1 represents World, 2 represents Sports, 3 represents Business and
    # 4 represents Sci/Tech.
    ag_df = pd.read_csv(path + '/train.csv', header=0, encoding='latin1')
    ag_df_test = pd.read_csv(path + '/test.csv', header=0, encoding='latin1')
    ag_df_concat = pd.concat([ag_df, ag_df_test], ignore_index=True)
    print(ag_df_concat.shape)
    print(ag_df_concat.columns)
    ag_df_concat = ag_df_concat.rename(columns={"Class Index": "labels"})
    ag_df_concat.groupby('labels')['labels'].value_counts()
    return ag_df_concat


def read_ag_news_split(path: str):
    # The first column is Class Index, the second column is Title and the third column is Description.
    # The class ids are numbered 1-4 where
    # 1 represents World, 2 represents Sports, 3 represents Business and
    # 4 represents Sci/Tech.
    ag_df = pd.read_csv(path + '/train.csv', header=0, encoding='latin1')
    ag_df_test = pd.read_csv(path + '/test.csv', header=0, encoding='latin1')
    print("SHAPE train ", ag_df.shape)
    print("SHAPE test ", ag_df_test.shape)
    print("TRAIN Cols ", ag_df.columns)
    ag_df= ag_df.rename(columns={"Class Index": "labels"})
    ag_df_test = ag_df_test.rename(columns={"Class Index": "labels"})
    print("TRAIN LABELS ", ag_df.groupby('labels')['labels'].value_counts())
    print("TEST LABELS ", ag_df_test.groupby('labels')['labels'].value_counts())
    print(ag_df.columns)
    return ag_df, ag_df_test

def read_bbc(path: str):
    bbc_df = pd.read_csv(path + '/bbc_data.csv', header=0, encoding='latin1')
    print(bbc_df.shape)
    print(bbc_df.columns)
    bbc_df.groupby('labels')['labels'].value_counts()
    return bbc_df

def read_news_category():
    news_category_df = pd.read_csv('News_Category_Dataset_v3.csv')
    print(news_category_df.shape)
    print(news_category_df.shape)
    print(news_category_df.columns)
    news_category_df = news_category_df.rename(columns={"category": "labels"})

    news_category_df.groupby('labels')['labels'].value_counts()
    return news_category_df

def clean_given_text(text):
    # Use BeautifulSoup to remove HTML tags, StringIO to avoid warning about URL format due to \ in text
    soup = BeautifulSoup(StringIO(str(text)), features='html.parser')
    clean_text = soup.get_text()

    #clean_text = replace_numeric_entities(clean_text)
    return clean_text

# trying to remove html character references and such - relevant for simpler encodings
def remove_character_references(df: pd.DataFrame, columns_to_apply_transforms_to: List[str]):
    for item in columns_to_apply_transforms_to:
        df[item] = df[item].apply(clean_given_text)
        df[item] = df[item].apply(html.unescape)
        df[item] = df[item].apply(urllib.parse.unquote)

def execute_replace_numeric_entities(text, pattern):
    # Replace all matches in the text with its char representation
    def replace_entity(match):
        # Convert the numeric part to an integer and then to the corresponding character
        return chr(int(match.group(1)))
    return re.sub(pattern, replace_entity, text)

def replace_numeric_entities(df: pd.DataFrame, columns_to_apply_transforms_to: List[str] ):
    for item in columns_to_apply_transforms_to:
        df[item] = df[item].apply(execute_replace_numeric_entities, pattern=r'#(\d+);')
    return df

def quot_occurences(df: pd.DataFrame, col: str):
    filtered_df = df[df[col].str.contains('quot;', na=False)]
    print(filtered_df.head(5)[col])
    print(filtered_df.shape)

def replace_quot(text):
    full_quote = r'FullQuote'
    quot_1=r'quot'
    quot= r'quot;'
    text = re.sub(quot, "\"", text)
    text = re.sub(full_quote, "\"", text)
    text = re.sub(quot_1, "\"", text)
    return text

def remove_quot_occurences(df: pd.DataFrame, cols_to_apply_transforms_to: List[str]):
    for item in cols_to_apply_transforms_to:
        df[item] = df[item].apply(replace_quot)

def initialize_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def get_stopwords_stemmer_lemmatizer():
    # define stop words
    stop_words = set(stopwords.words('english'))

    # using the stemmer from nltk - note that then nltk tokenizer is used too
    stemmer = PorterStemmer()

    # TODO: replace with spacy?
    lemmatizer = WordNetLemmatizer()
    return stop_words, stemmer, lemmatizer


def preprocess_text_nltk(text: str, stop_words: List[str]):
    # do it first on the complete df
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # tokenization by nltk tokenizer with stemming
    tokens = word_tokenize(text)

    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    return tokens


def preprocess_text_stemming(text: str, stemmer: PorterStemmer, stop_words: List[str]):
    tokens = preprocess_text_nltk(text, stop_words)
    # apply stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)


def preprocess_text_lemmatization(text: str, lemmatizer: WordNetLemmatizer, stop_words: List[str]):
    tokens = preprocess_text_nltk(text, stop_words)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_words)

def apply_preprocessing_stemming(train_df: pd.DataFrame, test_df: pd.DataFrame, col: str, stemmer: PorterStemmer):
    """
        Apply stemming with preprocessing to train and test data
    :param stemmer: stemmer for stemming
    :param train_df: train data
    :param test_df: test data
    :param col: column to apply stemming to
    :return: processed train data and test data
    """
    train_headlines_stem = [preprocess_text_stemming(text, stemmer) for text in train_df[col]]
    test_headlines_stem = [preprocess_text_stemming(text, stemmer) for text in test_df[col]]
    return train_headlines_stem, test_headlines_stem


def apply_preprocessing_lemmatization(train_df: pd.DataFrame, test_df: pd.DataFrame, col: str, lemmatizer: WordNetLemmatizer):
    """
        Apply lemmatization with preprocessing to train and test data
    :param lemmatizer: lemmatizer for lemmatizaton
    :param train_df: train data
    :param test_df: test data
    :param col: column to apply lemmatization to
    :return: processed train data and test data
    """
    train_headlines_lemmatization = [preprocess_text_lemmatization(doc, lemmatizer) for doc in train_df[col]]
    test_headlines_lemmatization = [preprocess_text_lemmatization(doc, lemmatizer) for doc in test_df[col]]
    return train_headlines_lemmatization, test_headlines_lemmatization

def initialize_vektorizer_tfidf():
    tfidf = TfidfVectorizer(max_features=5000,
                            #ngram_range=(2,2) # bigram vectorizing - bad for the model, default is skipgram
                           )
    return tfidf

def apply_tfidf(datasets: dict, tfidf: TfidfVectorizer):
    """
    Transform dataset with tfidf vectorization, note: keys in datasets expected to indicate whether this is train or test,
    train should always come first
    For datset dicts that contain more than one train/test set
    :param datasets:
    :param tfidf:
    :return:
    """
    for key, item in datasets.items():
        if key.__contains__('train'):
            datasets[key] = tfidf.fit_transform(datasets[key])
        if key.__contains__('test'):
            datasets[key] = tfidf.transform(datasets[key])
    return datasets

def apply_tfidf_single_train_test(datasets: dict, tfidf: TfidfVectorizer):
    """
    Transform dataset with tfidf vectorization, note: keys in datasets expected to indicate whether this is train or test,
    train should always come first
    expects datasets with ONE train/test set
    :param datasets:
    :param tfidf:
    :return:
    """

    datasets['train'] = tfidf.fit_transform(datasets['train'])
    datasets['test'] = tfidf.transform(datasets['test'])
    return datasets


def initialize_vektorizer_bow():
    bow = CountVectorizer(max_features=5000)
    return bow

def apply_bow(datasets: dict, bow: CountVectorizer):
    """
    Transform dataset with tfidf vectorization, note: keys in datasets expected to indicate whether this is train or test,
    train should always come first
    For datset dicts that contain more than one train/test set
    :param datasets:
    :param tfidf:
    :return:
    """
    for key, item in datasets.items():
        if key.__contains__('train'):
            datasets[key] = bow.fit_transform(datasets[key])
        if key.__contains__('test'):
            datasets[key] = bow.transform(datasets[key])
    return datasets

def apply_bow_single_train_test(datasets: dict, bow: CountVectorizer):
    """
    Transform dataset with tfidf vectorization, note: keys in datasets expected to indicate whether this is train or test,
    train should always come first
    expects datasets with ONE train/test set
    :param datasets:
    :param tfidf:
    :return:
    """

    datasets['train'] = bow.fit_transform(datasets['train'])
    datasets['test'] = bow.transform(datasets['test'])
    return datasets








"""
TODO: adapt to variable col names - work for ag-news dataset, but might need adaptions for others

"""

def clean_title_based_on_source(df):
    """
    This function cleans the 'Title' column of a DataFrame by removing any occurrence of the source text,
    if it is enclosed in parentheses within the title.

    For each row:
      - It retrieves the 'Title' and 'source' values.
      - If 'source' is valid (not null and a string), it creates a regex pattern that escapes
        the source text enclosed in parentheses.
      - It then removes the matched pattern from the Title.
      - Finally, it normalizes extra whitespace in the title and updates the row.

    The updated DataFrame is returned after applying this cleaning function to every row.
    """

    def process_row(row):
        title = row.get('Title', '')
        src = row.get('source', '')
        # print(title)
        # print(src)

        # Check if the source is a valid string.
        if pd.notnull(src) and src.count(" ") == 0:
            # Build the regex pattern to match the source when it appears enclosed in parentheses.
            pattern = re.escape(f"({src})").replace("\\", "")
            # print("PAttern", pattern)
            # Remove the pattern from the title.
            new_title = title.replace(pattern, "")
            # Replace multiple whitespace with a single space and remove leading/trailing spaces.
            new_title = re.sub(r'\s+', ' ', new_title).strip()
            # Update the Title in the row.
            row.loc['Title'] = new_title
            # print(row['Title'])

        return row

    # Apply the process_row function to every row in the DataFrame.
    return df.apply(process_row, axis=1)


def extract_source_from_description(df):
    """
    Extracts a source string from the 'Description' column of the DataFrame and updates the DataFrame accordingly.

    The function searches for separators in the beginning of the description as follows:
    - If a double hyphen (" -- ") is found within the first 35 characters, the part before it is
      considered the source and the rest is kept as the new description.
    - Otherwise, if a single hyphen (" - ") is found within the first 30 characters, the extraction
      is performed similarly.

    After processing, a new column 'source' is created (or updated) and the DataFrame columns are reordered
    so that 'source' appears as the second column.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Description' column to process.

    Returns:
        pd.DataFrame: Updated DataFrame with the extracted 'source' and modified 'Description'.
    """

    def process_row(row):
        desc = row['Description']
        source_str = None  # Default value if no source is found.
        new_desc = desc  # By default, keep the original description

        # Check if a double hyphen (" -- ") is found in the first 35 characters.
        if ' -- ' in desc[:35]:
            parts = desc.split('--', 1)  # Split only at the first occurrence.
            source_str = parts[0].strip()  # Extract and remove extra whitespace from source.
            new_desc = parts[1].strip()  # The remainder becomes the new description.
        # If the double hyphen pattern wasn't found, check for a single hyphen (" - ")
        # in the first 30 characters and do a similar extraction.
        elif ' - ' in desc[:30]:
            parts = desc.split('-', 1)
            source_str = parts[0].strip()
            new_desc = parts[1].strip()

        # Update the row with the extracted source and the cleaned description.
        row['source'] = source_str
        row['Description'] = new_desc
        return row

    # Apply the processing function to every row in the DataFrame.
    df = df.apply(process_row, axis=1)

    # Reorder columns so that 'source' is the second column.
    cols = df.columns.tolist()
    if 'source' in cols:
        cols.remove('source')
        cols.insert(1, 'source')
        df = df[cols]

    return df


def restructure_source_location(df):
    """
    Process the 'source' column of a DataFrame to potentially split it into two parts:
    the actual source and a location extracted from source text based on various patterns.

    Steps:
      1. Checks if a 'source' column exists. If not, prints an error and returns the DataFrame unchanged.
      2. Ensures a 'location' column exists by creating it with default NaN values if missing.
      3. Defines several regex patterns to match common source formatting,
         including adaptations:
           - Abbreviation pattern: if the 'source' text starts with terms like "Calif." or "Flo.",
             then that word is moved to the 'location' column.
           - Dot pattern: if the 'source' begins with a word ending in a dot (excluding first-level domain names)
             then that word is moved to the 'location' column.
      4. Applies these patterns to each row of the 'source' column using the process_text helper.
      5. For entries where a location was extracted, the 'source' column is updated to the new source value,
         and the new information is stored in the 'location' column.
      6. Finally, the columns are reordered so that 'location' appears as the third column.

    Args:
        df (pd.DataFrame): The input DataFrame that must contain a 'source' column.

    Returns:
        pd.DataFrame: The updated DataFrame with potentially restructured 'source' and extracted 'location'.
    """

    if 'source' not in df.columns:
        print("Column 'source' does not exist in this DataFrame.")
        return df

    if 'location' not in df.columns:
        df['location'] = np.nan

    pattern_paren = r"^\s*([A-Z ]+)\s*\(([^)]+)\)"
    pattern_comma = r"^\s*([A-Z ]+),\s*(.+)$"

    def process_text(text):
        """
        Applies regex checks to extract potential new 'source' and 'location' values.

        Matching order:
          1. Website pattern: e.g., 'SPACE.com' splits domain from following text.
          2. Abbreviation pattern: if the source starts with terms like "Calif." or "Flo."
             then that term is moved to 'location'.
          3. Adapted Dot pattern: if the source starts with a word ending with a '.'
             that is not immediately followed by a common first-level-domain,
             then the initial word is moved to 'location'.
          4. Combo pattern: If text contains a comma before parentheses, uses the parenthesized text as source.
          5. Parentheses pattern: Extracts text inside parentheses after uppercase letters.
          6. Comma pattern: Alternate extraction with a comma.
          7. Two capitalized words: Uses them as the location.
          8. Fallback: Extracts a single all-caps word (more than three characters) as location.
        """
        if isinstance(text, str):
            # Website pattern
            pattern_website = r"^\s*([A-Za-z]+\.(?:com|org|net|edu|gov|info|biz))\s*(.*)$"
            match_website = re.match(pattern_website, text)
            if match_website:
                source_val = match_website.group(1).strip()
                location_val = match_website.group(2).strip() if match_website.group(2) else None
                return source_val, location_val

            # Abbreviation pattern
            pattern_abbrev = r'^\s*((?:Calif\.|Flo\.))\s+(.*)$'
            match_abbrev = re.match(pattern_abbrev, text)
            if match_abbrev:
                return match_abbrev.group(2).strip(), match_abbrev.group(1).strip()

            # Adapted Dot pattern:
            # Matches a leading word ending with a period that is NOT followed by a first-level-domain.
            match_dot = re.match(r'^\s*([\w\-]+\.)(?!\s*(?:com|org|net|edu|gov|info|biz)\b)(.*)$', text)
            if match_dot:
                # Here, the part ending with the dot is moved to 'location'
                # while the rest of the text becomes the new source.
                return match_dot.group(2).strip(), match_dot.group(1).strip()

            # Combo pattern: text with parentheses at the end and a comma before.
            pattern_combo = r"^\s*(.+?)\s*\(([^()]+)\)\s*$"
            match_combo = re.match(pattern_combo, text)
            if match_combo:
                if "," in match_combo.group(1):
                    return match_combo.group(2).strip(), match_combo.group(1).strip()

            # Parentheses pattern.
            match = re.match(pattern_paren, text)
            if match:
                return match.group(2).strip(), match.group(1).strip()

            # Comma pattern.
            match2 = re.match(pattern_comma, text)
            if match2:
                return match2.group(2).strip(), match2.group(1).strip()

            # Look for two consecutive capitalized words.
            match_two = re.search(r'\b([A-Z]{2,})\s+([A-Z]{2,})\b', text)
            if match_two:
                location = f"{match_two.group(1)} {match_two.group(2)}"
                new_text = re.sub(re.escape(match_two.group(0)), '', text).strip()
                return new_text, location

            # Fallback: single all-caps word longer than three characters.
            candidates = re.findall(r'\b[A-Z]{4,}\b', text)
            if candidates:
                location = candidates[0]
                new_text = re.sub(r'\b' + re.escape(location) + r'\b', '', text).strip()
                return new_text, location
        return text, None

    processed = df["source"].apply(lambda x: process_text(x))
    new_sources = processed.apply(lambda t: t[0])
    locations = processed.apply(lambda t: t[1])

    mask = locations.notna()
    df.loc[mask, "source"] = new_sources[mask]
    df.loc[mask, "location"] = locations[mask]

    cols = df.columns.tolist()
    if "location" in cols:
        cols.remove("location")
        cols.insert(2, "location")
        df = df[cols]

    return df

CUSTOM_COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
    "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
    "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
    "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Côte d'Ivoire", "Cabo Verde", "Cambodia", "Cameroon",
    "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia",
    "Comoros", "Congo (Congo-Brazzaville)", "Costa Rica", "Croatia", "Cuba", "Cyprus",
    "Czechia (Czech Republic)", "Democratic Republic of the Congo", "Denmark", "Djibouti",
    "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador",
    "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini (fmr. 'Swaziland')",
    "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany",
    "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Holy See", "Honduras", "Hungary", "Iceland", "India", "Indonesia",
    "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan",
    "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", "Latvia",
    "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg",
    "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands",
    "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia",
    "Montenegro", "Morocco", "Mozambique", "Myanmar (formerly Burma)", "Namibia",
    "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria",
    "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau",
    "Palestine State", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines",
    "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa",
    "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia",
    "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
    "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka",
    "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Tajikistan", "Tanzania",
    "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia",
    "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States of America", "Uruguay",
    "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
]

def move_country_from_source_to_location(df, countries=CUSTOM_COUNTRIES):
    """
    Searches for any country names within the 'source' column based on a given list
    and moves it to the 'location' column.

    Args:
        df (pd.DataFrame): DataFrame with at least a 'source' column and an optional 'location' column.
        countries (list): List of country names to search for (default is custom_countries).

    Returns:
        pd.DataFrame: Updated DataFrame with the country name moved from 'source' to 'location'.
    """

    def process_row(row):
        source_text = row.get('source')
        location_text = row.get('location')

        # Skip processing if source_text is not valid.
        if pd.isna(source_text) or not isinstance(source_text, str):
            return row

        new_source = source_text
        found_country = None

        # Check for any country present in the custom list.
        for country in countries:
            pattern = r'\b' + re.escape(country) + r'\b'
            match = re.search(pattern, new_source, flags=re.IGNORECASE)
            if match:
                found_country = match.group(0)
                new_source = re.sub(pattern, "", new_source, flags=re.IGNORECASE).strip()
                break  # Stop after the first match

        # Update source with cleaned value (if empty, set to None)
        row['source'] = new_source if new_source != "" else None

        # Update location with the found country if applicable.
        if found_country:
            if pd.isna(location_text) or location_text.strip() == "":
                row['location'] = found_country
            else:
                row['location'] = f"{location_text.strip()} {found_country}"
        return row

    return df.apply(process_row, axis=1)

