import nltk
import spacy
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def read_speeches_csv(path):
    """Reads a CSV file containing speeches and returns a DataFrame."""
    df = pd.read_csv(path, encoding="utf-8")
    return df


def update_dataframe(df):
    """Updates the DataFrame based on the requirements listed in Part Two (a)."""
    # Create a new dataframe instead of overwriting
    df_copy = df.copy()

    # Rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’
    df_copy['party'] = df_copy['party'].replace('Labour (Co-op)', 'Labour')
    
    # Remove 'Speaker' value from the 'party' column
    df_copy = df_copy[df_copy['party'] != 'Speaker']

    # Fetch four most common party names
    four_most_common_party_names = df_copy['party'].value_counts().nlargest(4).index
    
    # Filter the DataFrame to keep only the rows with the top 4 party names
    df_filtered = df_copy[df_copy['party'].isin(four_most_common_party_names)]

    # Remove rows where 'speech_class' is not 'Speech'
    df_filtered = df_filtered[df_filtered['speech_class'] == 'Speech']
    
    # Remove rows where text in the 'speech' column is less than 1000 characters
    df_filtered = df_filtered[df_filtered['speech'].str.len() >= 1000]
    
    print(f"Number of rows: {df_filtered.shape[0]}")
    print(f"Number of columns: {df_filtered.shape[1]}")
    
    return df_filtered


def vectorize_speeches(df, ngram_range=(1, 1)):
    """Vectorizes speeches and splits into train/test sets. 
    Assigned (1, 1) as the default value for ngram_range parameter as the vectorizer internally 
    considers only unigrams when ngram_range is not specified.
    """
    vectorizer = TfidfVectorizer(stop_words='english', 
                                 max_features=3000,
                                 ngram_range=ngram_range)
    
    tfidf_matrix = vectorizer.fit_transform(df['speech'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix,
        df['party'],
        test_size=0.2,
        random_state=26,
        stratify=df['party']
        )
    
    #print(X_train.shape, X_test.shape)
    #print(y_train.value_counts(normalize=True))
    return X_train, X_test, y_train, y_test
    
    
def train_randomforest_svm_models(X_train, X_test, y_train, y_test):
    """Trains RandomForest and SVM models using the training data."""
    
    # Train Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=300, random_state=26)
    # Fit the model to the training data
    rf_clf.fit(X_train, y_train)
    # Predict on the test set for ‘party’ value
    rf_y_pred = rf_clf.predict(X_test)
    # Print the F1 score
    print("Random Forest classifier \n\nF1 score:", f1_score(y_test, rf_y_pred, average='macro'))
    # Print the classification report
    print("\nClassification Report:\n", classification_report(y_test, rf_y_pred, zero_division=0))
    
    # Train SVM Classifier with linear kernel
    svm_clf = SVC(kernel='linear', random_state=26)
    # Fit the model to the training data
    svm_clf.fit(X_train, y_train)
    # Predict on the test set for ‘party’ value
    svm_y_pred = svm_clf.predict(X_test)
    # Print the F1 score
    print("SVM classifier \n\nF1 score:", f1_score(y_test, svm_y_pred, average='macro'))
    # Print the classification report
    print("\nClassification Report:\n", classification_report(y_test, svm_y_pred))
    

def custom_tokenizer_spacy(text):
    """Custom tokenizer function for the vectorizer."""
    doc = nlp(text.lower())
    clean_tokens = []
    allowed_pos_list = ['NOUN', 'ADJ', 'VERB', 'ADV']
        
    for token in doc:
        # Skip tokens that are stop words, punctuation, spaces, non-alphabetic, or not in the allowed POS list
        if (
            token.is_stop
            or token.is_punct
            or token.is_space
            or not token.is_alpha
            or token.pos_ not in allowed_pos_list
        ):
            continue
        # Append the lemmatized token to the clean_tokens list
        clean_tokens.append(token.lemma_)
    return clean_tokens


def custom_tokenizer_nltk(text):
    # Custom tokenizer function for the vectorizer using NLTK.
    tokens = word_tokenize(text)
    clean_tokens = []
    for token in tokens:
       token = token.lower()
       if token.isalpha() and token not in stop_words:
        lemma = lemmatizer.lemmatize(token)
        if len(token) > 1:
            clean_tokens.append(lemma)
    return clean_tokens


def vectorize_speeches_with_custom_tokenizer(df, num_features, custom_tokenizer, ngram_range=(1, 1)):
    """Vectorizes speeches and splits into train/test sets. 
    Assigned (1, 1) as the default value for ngram_range parameter as the vectorizer internally 
    considers only unigrams when ngram_range is not specified.
    """
    vectorizer = TfidfVectorizer(max_features=num_features,
                                 ngram_range=ngram_range,
                                 tokenizer=custom_tokenizer)
    tfidf_matrix = vectorizer.fit_transform(df['speech'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix,
        df['party'],
        test_size=0.2,
        random_state=26,
        stratify=df['party']
        )

    return X_train, X_test, y_train, y_test


def best_classification_performance_with_custom_tokenizer(df, custom_tokenizer):
    """Finds the best classification performance using a custom tokenizer."""
    feature_ranges = [2000, 2500, 3000]
    classifiers_dict = {
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=26),
        "SVM": lambda: SVC(kernel='linear', random_state=26)
    }
    perf_results = []
    
    for num_features in feature_ranges:
        print(f"\n\nTesting with max_features={num_features}\n")
        X_train, X_test, y_train, y_test = vectorize_speeches_with_custom_tokenizer(df, num_features, custom_tokenizer, ngram_range=(1, 3))
        feature_count = X_train.shape[1]
        
        for clf_name, clf_func in classifiers_dict.items():
            clf = clf_func()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1_macro_score = f1_score(y_test, y_pred, average='macro')
            report = classification_report(y_test, y_pred, zero_division=0)
            perf_results.append({
                "Classifier": clf_name,
                "Max Features": num_features,
                "Feature Count": feature_count,
                "F1 Score": f1_macro_score,
                "Classification Report": report
            })
            
            if clf_name == "SVM":
                print(f"{clf_name} with max_features={num_features}:\nF1 Score: {f1_macro_score}\n{report}\n")

    return perf_results

if __name__ == "__main__":
    """
    Main method
    """
    path = Path.cwd() / "p2-texts" / "hansard40000.csv"
    df = read_speeches_csv(path)
    df_filtered = update_dataframe(df)
    X_train, X_test, y_train, y_test = vectorize_speeches(df_filtered)
    train_randomforest_svm_models(X_train, X_test, y_train, y_test)
    # Calling vectorize_speeches function with ngram_range=(1, 3) to include unigrams, bigrams, and trigrams.
    X_train_new, X_test_new, y_train_new, y_test_new = vectorize_speeches(df_filtered, ngram_range=(1, 3))
    train_randomforest_svm_models(X_train_new, X_test_new, y_train_new, y_test_new)
    # spacy_perf_results = best_classification_performance_with_custom_tokenizer(df_filtered, custom_tokenizer=custom_tokenizer_spacy)
    nltk_perf_results = best_classification_performance_with_custom_tokenizer(df_filtered, custom_tokenizer=custom_tokenizer_nltk)
    




