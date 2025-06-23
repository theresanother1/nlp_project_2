import os
import re
import html
import urllib.parse
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, TruncatedSVD, PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import os

import matplotlib.pyplot as plt

LABELS = {
    1: "World",
    2: "Sports", 
    3: "Business",
    4: "Science/Tech"
}

custom_countries = [
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

class DataProcessor:
    def __init__(self, train_fp, test_fp):
        self.train_fp = train_fp
        self.test_fp = test_fp
        self.train_cleaned_fp = self._get_cleaned_fp(train_fp)
        self.test_cleaned_fp = self._get_cleaned_fp(test_fp)
        self.df_dict = {}

    def _get_cleaned_fp(self, fp):
        base, ext = os.path.splitext(fp)
        return f"{base}_cleaned.csv"

    def load_and_clean(self):
        if os.path.exists(self.train_cleaned_fp) and os.path.exists(self.test_cleaned_fp):
            print("Loading cleaned data files...")
            self.df_dict['train'] = pd.read_csv(self.train_cleaned_fp)
            self.df_dict['test'] = pd.read_csv(self.test_cleaned_fp)
        else:
            print("Processing and saving cleaned data files...")
            self.df_dict['train'] = pd.read_csv(self.train_fp)
            self.df_dict['test'] = pd.read_csv(self.test_fp)
            self.df_dict['train']["Class Index"] = self.df_dict['train']["Class Index"].map(LABELS)
            self.df_dict['test']["Class Index"] = self.df_dict['test']["Class Index"].map(LABELS)
            for key in self.df_dict:
                df = self.df_dict[key]
                df = self.full_text_transform(df, "Description")
                df = self.extract_source_from_description(df)
                df = self.clean_title_based_on_source(df)
                df = self.restructure_source_location(df)
                df = self.move_country_from_source_to_location(df)
                self.df_dict[key] = df
            self.df_dict['train'].to_csv(self.train_cleaned_fp, index=False)
            self.df_dict['test'].to_csv(self.test_cleaned_fp, index=False)
        return self.df_dict['train'], self.df_dict['test']

    def full_text_transform(self, df, column, numeric_entity_pattern=r'#(\d+);'):
        def clean_and_replace(text):
            soup = BeautifulSoup(StringIO(str(text)), features='html.parser')
            clean_text = soup.get_text()
            clean_text = html.unescape(clean_text)
            clean_text = urllib.parse.unquote(clean_text)
            def replace_entity(match):
                return chr(int(match.group(1)))
            clean_text = re.sub(numeric_entity_pattern, replace_entity, clean_text)
            clean_text = re.sub(r'\b(http|www|href|aspx|com|org|net|edu|gov|info|biz)\b', '', clean_text, flags=re.IGNORECASE)
            clean_text = re.sub(
                r'\b(Monday|Mon|Tuesday|Tue|Tues|Wednesday|Wed|Thursday|Thu|Thurs|Friday|Fri|Saturday|Sat|Sunday|Sun)\b',
                '', clean_text, flags=re.IGNORECASE)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            return clean_text
        df[column] = df[column].apply(clean_and_replace)
        return df

    def extract_source_from_description(self, df):
        def process_row(row):
            desc = row['Description']
            source_str = None
            new_desc = desc
            if ' -- ' in desc[:35]:
                parts = desc.split('--', 1)
                source_str = parts[0].strip()
                new_desc = parts[1].strip()
            elif ' - ' in desc[:30]:
                parts = desc.split('-', 1)
                source_str = parts[0].strip()
                new_desc = parts[1].strip()
            row['source'] = source_str
            row['Description'] = new_desc
            return row
        df = df.apply(process_row, axis=1)
        cols = df.columns.tolist()
        if 'source' in cols:
            cols.remove('source')
            cols.insert(1, 'source')
            df = df[cols]
        return df

    def clean_title_based_on_source(self, df):
        def process_row(row):
            title = row.get('Title', '')
            src = row.get('source', '')
            if pd.notnull(src) and src.count(" ") == 0:
                pattern = re.escape(f"({src})").replace("\\", "")
                new_title = title.replace(pattern, "")
                new_title = re.sub(r'\s+', ' ', new_title).strip()
                row.loc['Title'] = new_title
            return row
        return df.apply(process_row, axis=1)

    def move_country_from_source_to_location(self, df, countries=custom_countries):
        def process_row(row):
            source_text = row.get('source')
            location_text = row.get('location')
            if pd.isna(source_text) or not isinstance(source_text, str):
                return row
            new_source = source_text
            found_country = None
            for country in countries:
                pattern = r'\b' + re.escape(country) + r'\b'
                match = re.search(pattern, new_source, flags=re.IGNORECASE)
                if match:
                    found_country = match.group(0)
                    new_source = re.sub(pattern, "", new_source, flags=re.IGNORECASE).strip()
                    break
            row['source'] = new_source if new_source != "" else None
            if found_country:
                if pd.isna(location_text) or location_text.strip() == "":
                    row['location'] = found_country
                else:
                    row['location'] = f"{location_text.strip()} {found_country}"
            return row
        return df.apply(process_row, axis=1)

    def restructure_source_location(self, df):
        if 'source' not in df.columns:
            print("Column 'source' does not exist in this DataFrame.")
            return df
        if 'location' not in df.columns:
            df['location'] = np.nan
        pattern_paren = r"^\s*([A-Z ]+)\s*\(([^)]+)\)"
        pattern_comma = r"^\s*([A-Z ]+),\s*(.+)$"
        def process_text(text):
            if isinstance(text, str):
                pattern_website = r"^\s*([A-Za-z]+\.(?:com|org|net|edu|gov|info|biz))\s*(.*)$"
                match_website = re.match(pattern_website, text)
                if match_website:
                    source_val = match_website.group(1).strip()
                    location_val = match_website.group(2).strip() if match_website.group(2) else None
                    return source_val, location_val
                pattern_abbrev = r'^\s*((?:Calif\.|Flo\.))\s+(.*)$'
                match_abbrev = re.match(pattern_abbrev, text)
                if match_abbrev:
                    return match_abbrev.group(2).strip(), match_abbrev.group(1).strip()
                match_dot = re.match(r'^\s*([\w\-]+\.)(?!\s*(?:com|org|net|edu|gov|info|biz)\b)(.*)$', text)
                if match_dot:
                    return match_dot.group(2).strip(), match_dot.group(1).strip()
                pattern_combo = r"^\s*(.+?)\s*\(([^()]+)\)\s*$"
                match_combo = re.match(pattern_combo, text)
                if match_combo:
                    if "," in match_combo.group(1):
                        return match_combo.group(2).strip(), match_combo.group(1).strip()
                match = re.match(pattern_paren, text)
                if match:
                    return match.group(2).strip(), match.group(1).strip()
                match2 = re.match(pattern_comma, text)
                if match2:
                    return match2.group(2).strip(), match2.group(1).strip()
                match_two = re.search(r'\b([A-Z]{2,})\s+([A-Z]{2,})\b', text)
                if match_two:
                    location = f"{match_two.group(1)} {match_two.group(2)}"
                    new_text = re.sub(re.escape(match_two.group(0)), '', text).strip()
                    return new_text, location
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
        df["location"] = df["location"].astype('object')
        return df

class ExperimentRunner:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

    def run_nmf_experiments(self):
        X_text = self.train_df['Description']
        y = self.train_df['Class Index']
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X_all = vectorizer.fit_transform(X_text)
        le = LabelEncoder()
        y_all_enc = le.fit_transform(y)
        X_train, X_val, y_train, y_val = train_test_split(X_all, y_all_enc, test_size=0.2, random_state=42, stratify=y_all_enc)
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Linear SVC": LinearSVC(max_iter=1000, random_state=42),
            "Naive Bayes": GaussianNB()
        }
        results = []
        for n in [8, 16, 32, 64, 128]:
            nmf_model = NMF(n_components=n, random_state=42, max_iter=500)
            W_train = nmf_model.fit_transform(X_train)
            W_val = nmf_model.transform(X_val)
            print(f"\nNMF n_components={n}")
            for name, clf in classifiers.items():
                clf.fit(W_train, y_train)
                y_pred = clf.predict(W_val)
                acc = accuracy_score(y_val, y_pred)
                print(f"  {name}: Accuracy={acc:.4f}")
                results.append({"n_components": n, "Classifier": name, "Accuracy": acc})
        results_df = pd.DataFrame(results)
        return results_df

    def run_lsa_experiments(self):
        X_text = self.train_df['Description']
        y = self.train_df['Class Index']
        max_features_list = [500, 1000, 2000]
        n_components_list = [50, 100, 200]
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Linear SVC": LinearSVC(max_iter=1000, random_state=42),
            "Naive Bayes": GaussianNB()
        }
        results = []
        for max_features in max_features_list:
            for n_components in n_components_list:
                print(f"\n[INFO] Vectorizing with max_features={max_features}, LSA n_components={n_components}")
                vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
                X = vectorizer.fit_transform(X_text)
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                X_reduced = svd.fit_transform(X)
                X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)
                for name, clf in classifiers.items():
                    print(f"  [INFO] Training {name} ...", end="")
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_val)
                    acc = (y_pred == y_val).mean()
                    print(f" done. Accuracy={acc:.4f}")
                    results.append({
                        "max_features": max_features,
                        "n_components": n_components,
                        "Classifier": name,
                        "Accuracy": acc
                    })
        results_df = pd.DataFrame(results)
        return results_df

    def run_bow_tfidf_experiments(self):
        X_text = self.train_df['Description']
        y = self.train_df['Class Index']
        bow_params = {'max_features': [500, 1000, 2000], 'ngram_range': [(1,1), (1,2)]}
        tfidf_params = {'max_features': [1000, 2000, 3000], 'ngram_range': [(1,1), (1,2)]}
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Linear SVC": LinearSVC(max_iter=1000, random_state=42),
            "Multinomial NB": MultinomialNB()
        }
        results = []
        for max_feat in bow_params['max_features']:
            for ngram in bow_params['ngram_range']:
                bow_vectorizer = CountVectorizer(stop_words='english', max_features=max_feat, ngram_range=ngram)
                X_bow = bow_vectorizer.fit_transform(X_text)
                X_train, X_val, y_train, y_val = train_test_split(X_bow, y, test_size=0.2, random_state=42, stratify=y)
                for name, clf in classifiers.items():
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_val)
                    acc = accuracy_score(y_val, y_pred)
                    results.append({
                        "Method": "BoW",
                        "max_features": max_feat,
                        "ngram_range": str(ngram),
                        "Classifier": name,
                        "Accuracy": acc
                    })
        for max_feat in tfidf_params['max_features']:
            for ngram in tfidf_params['ngram_range']:
                tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=max_feat, ngram_range=ngram)
                X_tfidf = tfidf_vectorizer.fit_transform(X_text)
                X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
                for name, clf in classifiers.items():
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_val)
                    acc = accuracy_score(y_val, y_pred)
                    results.append({
                        "Method": "TF-IDF",
                        "max_features": max_feat,
                        "ngram_range": str(ngram),
                        "Classifier": name,
                        "Accuracy": acc
                    })
        results_df = pd.DataFrame(results)
        return results_df

    def run_pca_experiment(self):
        X_text = self.train_df['Description']
        y = self.train_df['Class Index']
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(X_text).toarray()
        n_topics = 4
        pca = PCA(n_components=n_topics, random_state=42)
        X_pca = pca.fit_transform(X)
        feature_names = vectorizer.get_feature_names_out()
        for i, component in enumerate(pca.components_):
            top_indices = component.argsort()[-10:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            print(f"Topic {i+1}: {' '.join(top_words)}")
        X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Linear SVC": LinearSVC(max_iter=1000, random_state=42),
            "Naive Bayes": GaussianNB()
        }
        for name, clf in classifiers.items():
            print(f"\nClassifier: {name}")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            print(classification_report(y_val, y_pred))

if __name__ == "__main__":
    
    train_fp = os.path.join('data', 'train.csv')
    test_fp = os.path.join('data', 'test.csv')
    
    processor = DataProcessor(train_fp, test_fp)
    train_df, test_df = processor.load_and_clean()
    print("\nCleaned Train DataFrame Shape:", train_df.shape)
    print("Cleaned Train DataFrame:")
    print(train_df.head().to_markdown())
    print("\n" + "="*50 + "\n")
    print("Cleaned Test DataFrame Shape:", test_df.shape)
    print("\nCleaned Test DataFrame:")
    print(test_df.head().to_markdown())

    runner = ExperimentRunner(train_df, test_df)
    nmf_results = runner.run_nmf_experiments()
    print(nmf_results)
    lsa_results = runner.run_lsa_experiments()
    print(lsa_results)
    bow_tfidf_results = runner.run_bow_tfidf_experiments()
    print(bow_tfidf_results)
    runner.run_pca_experiment()
    print("PCA experiment completed.")
    plt.figure(figsize=(10, 6))
    plt.plot(nmf_results['n_components'], nmf_results['Accuracy'], marker='o', label='NMF Accuracy')
    plt.plot(lsa_results['n_components'], lsa_results['Accuracy'], marker='x', label='LSA Accuracy')
    plt.plot(bow_tfidf_results['max_features'], bow_tfidf_results['Accuracy'], marker='s', label='BoW/TF-IDF Accuracy')
    plt.xlabel('Number of Components / Features')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs Number of Components / Features')
    plt.legend()
    plt.grid()