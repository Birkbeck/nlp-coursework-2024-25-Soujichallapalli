import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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


def vectorize_speeches(df):
    """Vectorizes speeches and splits into train/test sets."""
    vectorizer = TfidfVectorizer(stop_words='english', 
                                 max_features=3000)
    
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
    
    
if __name__ == "__main__":
    """
    Main method
    """
    path = Path.cwd() / "p2-texts" / "hansard40000.csv"
    df = read_speeches_csv(path)
    df_filtered = update_dataframe(df)
    X_train, X_test, y_train, y_test = vectorize_speeches(df_filtered)



