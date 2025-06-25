import pandas as pd
from pathlib import Path


def read_speeches_csv(path):
    """Reads a CSV file containing speeches and returns a DataFrame.

    Args:
        path (Path): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the speeches.
    """
    df = pd.read_csv(path, encoding="utf-8")
    return df

def update_dataframe(df):
    """Updates the DataFrame based on the requirements listed in Part Two (a)."""
    # Rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')

    # Fetch four most common party names
    four_most_common_party_names = df['party'].value_counts().nlargest(4).index
    
    # Remove 'Speaker' value from the 'party' column
    four_most_common_party_names = four_most_common_party_names[four_most_common_party_names != 'Speaker']
    
    # Filter the DataFrame to keep only the rows with the top 4 party names
    df = df[df['party'].isin(four_most_common_party_names)]

    # Remove rows where 'speech_class' is not 'Speech'
    df = df[df['speech_class'] == 'Speech']
    
    # Remove rows where text in the 'speech' column is less than 1000 characters
    df = df[df['speech'].str.len() >= 1000]
    
    num_rows, num_cols = df.shape
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")


if __name__ == "__main__":
    """
    Main method
    """
    path = Path.cwd() / "p2-texts" / "hansard40000.csv"
    print(path)
    df = read_speeches_csv(path)
    print(df.head())
    print("\n")
    update_dataframe(df)



