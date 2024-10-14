import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
from bs4 import BeautifulSoup
import requests
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('cmudict', quiet=True)

# Load CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

def count_syllables(word):
    """Count syllables in a word using CMU Pronouncing Dictionary and fallback to regex method."""
    word = word.lower()
    if word in cmu_dict:
        return max([len([y for y in x if y[-1].isdigit()]) for x in cmu_dict[word]])
    else:
        # Fallback method
        count = 0
        vowels = 'aeiouy'
        if word and word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        return count if count > 0 else 1  # Ensure at least one syllable

def is_complex_word(word):
    """Check if a word is complex (more than two syllables)."""
    return count_syllables(word) > 2 

def is_personal_pronoun(word):
    """Check if a word is a personal pronoun, considering case."""
    personal_pronouns = {'i', 'we', 'me', 'us', 'my', 'our', 'mine', 'ours'}
    return word.lower() in personal_pronouns and not word.isupper() 

def load_words(file_path):
    """Load words from a file, handling different encodings."""
    words = set()
    encodings = ['utf-8', 'iso-8859-1', 'latin-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                words.update(word.strip().lower() for word in file if word.strip())
            break
        except UnicodeDecodeError:
            continue
    return words

def load_stopwords(stopword_files):
    """Load custom stopwords from multiple files."""
    stop_words = set()
    for file in stopword_files:
        stop_words.update(load_words(file))
    return stop_words

def get_article_text(url, url_id):
    """Scrape article text from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        heading = soup.find('h1', class_='entry-title') #specify the specific class with regards to your html script
        content = soup.find(class_='td-post-content tagdiv-type') #specify the class specific to you
        extracted_text = []
        if heading:
            extracted_text.append(heading.get_text(strip=True))
        if content:
            for tag in content.find_all(['p', 'ol', 'ul', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                extracted_text.append(tag.get_text(strip=True))
        full_text = '\n'.join(extracted_text)
        
        # Save the extracted article text to a file
        with open(f"{url_id}.txt", "w", encoding='utf-8') as f:
            f.write(full_text)
        
        return full_text
    except Exception as e:
        print(f"Error processing {url_id}: {str(e)}")
        return ""

def analyze_text(text, positive_words, negative_words, stop_words):
    """Perform text analysis on the given text."""
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Clean words (remove punctuation)
    cleaned_words = [word for word in words if word.isalnum()]
    
    # Remove stop words (check lower, original, and upper case)
    words_without_stopwords = [word for word in cleaned_words if word.lower() not in stop_words and word not in stop_words and word.upper() not in stop_words]
    
    positive_score = sum(1 for word in words_without_stopwords if word.lower() in positive_words)
    negative_score = sum(1 for word in words_without_stopwords if word.lower() in negative_words)
    
    total_words = len(words_without_stopwords)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (total_words + 0.000001)
    
    avg_sentence_length = len(words) / len(sentences)
    complex_words = [word for word in words_without_stopwords if is_complex_word(word)]
    complex_word_count = len(complex_words)
    percentage_complex_words = complex_word_count / total_words if total_words > 0 else 0
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    
    syllable_count = sum(count_syllables(word) for word in words_without_stopwords)
    syllable_per_word = syllable_count / total_words if total_words > 0 else 0
    
    personal_pronouns = sum(1 for word in words if is_personal_pronoun(word))
    avg_word_length = sum(len(word) for word in words_without_stopwords) / total_words if total_words > 0 else 0
    
    return {
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score,
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_sentence_length,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': total_words,
        'SYLLABLE PER WORD': syllable_per_word,
        'PERSONAL PRONOUNS': personal_pronouns,
        'AVG WORD LENGTH': avg_word_length
    }

def process_articles(input_file, output_file, positive_words_file, negative_words_file, stopword_files):
    """Main function to process articles and generate analysis."""
    # Load input data
    input_data = pd.read_excel(input_file)
    
    # Load word lists
    print("Loading word lists...")
    positive_words = load_words(positive_words_file)
    negative_words = load_words(negative_words_file)
    stop_words = load_stopwords(stopword_files)
    
    # Scrape and analyze articles
    results = []
    print("Scraping and analyzing articles...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(get_article_text, row['URL'], row['URL_ID']): row for _, row in input_data.iterrows()}
        for future in tqdm(as_completed(future_to_url), total=len(input_data), desc="Progress"):
            row = future_to_url[future]
            try:
                text = future.result()
                if text:
                    analysis = analyze_text(text, positive_words, negative_words, stop_words)
                    results.append({**row.to_dict(), **analysis})
                else:
                    print(f"No content found for {row['URL_ID']}")
            except Exception as e:
                print(f"Error processing {row['URL_ID']}: {str(e)}")
    
    # Create and save output
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to '{output_file}'")

if __name__ == "__main__":
    input_file = 'Input.xlsx'
    output_file = 'Output.csv'
    positive_words_file = 'MasterDictionary/positive-words.txt'
    negative_words_file = 'MasterDictionary/negative-words.txt'
    
    # List of custom stopwords files I have used those for example.
    stopword_files = [
        "StopWords/StopWords_Auditor.txt",
        "StopWords/StopWords_Currencies.txt",
        "StopWords/StopWords_DatesandNumbers.txt",
        "StopWords/StopWords_Generic.txt",
        "StopWords/StopWords_GenericLong.txt",
        "StopWords/StopWords_Geographic.txt",
        "StopWords/StopWords_Names.txt"
    ]
    
    process_articles(input_file, output_file, positive_words_file, negative_words_file, stopword_files)
