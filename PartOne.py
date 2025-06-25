#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import glob
import math
import nltk
import spacy
import pickle
import pandas as pd
from pathlib import Path
from collections import Counter

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    words = nltk.word_tokenize(text)
    # Only consider alphabetic tokens and convert to lowercase
    clean_words = [word.lower() for word in words if word.isalpha()]
    total_sentences = len(nltk.sent_tokenize(text))
    total_syllables = 0
    if len(clean_words) == 0 or total_sentences == 0:
        return 0.0    
    for word in clean_words:
        total_syllables += count_syl(word, d)
    fk = (0.39 * (len(clean_words) / total_sentences)) + (11.8 * (total_syllables / len(clean_words))) - 15.59
    return fk


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    num_syllables = 0    
    if word in d:
        # Iterate through the first available phoneme list
        for phoneme in d[word][0]:
            # The number of syllables in a pronunciation is equal to the number of stressed vowels 
            if phoneme[-1].isdigit():
                num_syllables += 1
    else:
        previous_char_vowel = False

        # Count syllables by counting vowel clusters
        for char in word:
            # Check if the character is a vowel. 'Y' is also considered a vowel here.
            if char in "aeiouy" and (not previous_char_vowel):
                num_syllables += 1
                previous_char_vowel = True
            else:
                previous_char_vowel = False
    
        # Adjust for silent 'e' endings
        if len(word) > 1 and word.endswith('e'):
            num_syllables -= 1  
    # Ensure at least one syllable is counted
    return max(num_syllables, 1)


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    data = []
    # Creates a pandas dataframe with the following columns: text, title, author,year
    for txt_file in path.glob("*.txt"):
        title, author, year = txt_file.stem.split("-")
        title = title.replace("_", " ").strip()
        with open(txt_file, "r", encoding='utf-8') as file_handler:
            text = file_handler.read()
            data.append({
                "text": text,
                "title": title,
                "author": author,
                "year": year
            })
    df = pd.DataFrame(data)
    # Sorts the dataframe by the year column before returning it, resetting or ignoring the dataframe index.
    sorted_df = df.sort_values(by='year', ascending=True).reset_index(drop=True) 
    return sorted_df


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes
    the resulting  DataFrame to a pickle file"""
    # Ensure the store_path exists
    store_path.mkdir(parents=True, exist_ok=True)
    # Stores the parsed documents in a new column called 'parsed' in the DataFrame
    df['parsed'] = df['text'].apply(lambda x: process_text_in_sections(x, nlp))

    # Serialise the DataFrame using pickle format
    with open(store_path / out_name, 'wb') as f:
        pickle.dump(df, f)

    # Alternatively, we can use pandas 'to_pickle' method to serialise the DataFrame
    # df.to_pickle(store_path / out_name)
    
    # Load the DataFrame
    df = pd.read_pickle(store_path / out_name)
    
    # Return the dataframe
    return df


def process_text_in_sections(text, nlp, section_size=100000):
    """Processes the text in sections to avoid memory issues with large texts."""
    if len(text) <= nlp.max_length:
        return nlp(text)
    else:
        sections = [text[i:i + section_size] for i in range(0, len(text), section_size)]
        # Concatenate processed sections' texts
        concatenated_doc = ' '.join([nlp(section).text for section in sections])
        return nlp(concatenated_doc)


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = []
    tokens.extend(nltk.word_tokenize(text))
    tokens = [token.lower() for token in tokens if token.isalpha()]
    if len(tokens) == 0:
        return 0.0  # Avoid division by zero error when there are no tokens
    ttr = len(set(tokens)) / len(tokens)
    return ttr


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb, top_elements=10):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list ordered by their Pointwise Mutual Information."""
    subjects_list = []
    verb_counter = 0
    subject_counter = Counter()
    subject_verb_counter = Counter()
    total_tokens = len(doc)

    for token in doc:
        if token.lemma_ == target_verb and token.pos_ == "VERB":
            verb_counter += 1
            subject = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
            for sub in subject:
                subject_text = sub.text.lower()
                subjects_list.append(subject_text)
                subject_verb_counter[subject_text] += 1
        if token.dep_ in ("nsubj", "nsubjpass"):
            subject_counter[token.text.lower()] += 1

    # Calculate PMI for each subject using th formula log((P(subject, verb) / (P(subject) * P(verb))))
    # where P(subject, verb) is the joint probability of the subject and verb,
    # P(subject) is the probability of the subject, and P(verb) is the probability of the verb.
    pmi_scores_dict = {}
    for subject, count in subject_verb_counter.items():
        prob_subject = subject_counter[subject] / total_tokens
        prob_verb = verb_counter / total_tokens
        joint_prob_subject_verb = count / total_tokens
        pmi_val = math.log2(joint_prob_subject_verb / (prob_subject * prob_verb))
        pmi_scores_dict[subject] = pmi_val

    # Sort subjects by PMI score
    sorted_subjects = sorted(pmi_scores_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_subjects[:top_elements]



def subjects_by_verb_count(doc, verb, count=10):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = []
    for token in doc:
        if token.lemma_ == "hear" and token.pos_ == "VERB":
            subject = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
            for sub in subject:
                subjects.append(sub.text.lower())
    most_common_subjects = Counter(subjects).most_common(count)
    return most_common_subjects            


def adjective_counts(df, count=10):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    adjectives = []
    for i, row in df.iterrows():
        doc = row["parsed"]
        adjectives = [token.text.lower() for token in doc if token.pos_ == "ADJ"]
        most_common_adjectives = Counter(adjectives).most_common(count)
        print(most_common_adjectives)
        print("\n")


def common_objects(doc, count):
    """Returns a list of the ten most common syntactic objects overall in the text"""
    objects = []
    objects = [token.text.lower() for token in doc if token.dep_ == "dobj"]
    most_common_objects = Counter(objects).most_common(count)
    return most_common_objects


if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path)
    print(df.head())
    nltk.download("cmudict")
    df = parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    print(adjective_counts(df))
    
    for i, row in df.iterrows():
        print(row["title"])
        print(common_objects(row["parsed"], 10))
        print("\n")
            
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")