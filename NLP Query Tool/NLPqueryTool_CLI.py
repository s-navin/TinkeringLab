# # NLP Query-Response Tool
# # author: Navin S 

# imports 
import os

from bs4 import BeautifulSoup

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Parse HTML Text
def parse_html_files(folder_path):
    """ 
    Parses HTML text and extracts the question-response pairs from the FAQ (HTML format) documents. 
    
    Input Arguments: 
        folder_path (str): path to folder containing FAQs documents in HTML format
    
    Returns: 
        faq_data (list): list of parsed query-response pairs, extracted from the HTML files 
    """
    faq_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.html'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                
                #key_question
                query_body = soup.find('h1', class_='article-title')
                if (not query_body): 
                    key_question = "Apologies! HTML content glitch encountered"
                    answer = "Apologies! HTML content glitch encountered"
                    continue
                key_question = soup.find('h1', class_='article-title').get('title')
                
                article_body = soup.find('div', class_='article-body')
                if (not article_body): 
                    answer = "Apologies! Could not find a relevant answer to your query!"
                    continue
                answer = article_body.get_text(separator='\n', strip=True)
                                
                faq_data.append((key_question, answer))
                    
    return faq_data

# Text Pre-processing
def preprocess_text(text):
    """ 
    Tokenizes and pre-processes the text data, and eliminates stop-words.     
    
    Input Arguments: 
        text (str): text to be tokenized and pre-processed 
    
    Returns: 
        tokens (list): list of tokens 
    """
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    return tokens

# Text Representation
def vectorize_text(faq_data):
    """ 
    Converts text data into word vectors using TF-IDF.     
    
    Input Arguments: 
        faq_data (list): list of parsed query-response pairs, extracted from the FAQ (HTML) documents 
    
    Returns: 
        vectorizer (object)  : TfidfVectorizer object
        tfidf_matrix (object): sparse matrix object        
    """    
    questions = [item[0] for item in faq_data]
    vectorizer = TfidfVectorizer(tokenizer=preprocess_text, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(questions)
    return vectorizer, tfidf_matrix

# Query Matching
def find_best_match(query, vectorizer, X, faq_data):
    """ 
    Matches user queries with the key questions in the FAQ (HTML) documents using cosine similarity, 
    and returns a suitable response
    
    Input Arguments: 
        query (str)         : user-query
        vectorizer (object) : TfidfVectorizer object
        X (object)          : sparse tfidf_matrix object        
        faq_data (list)     : list of parsed query-response pairs, extracted from the FAQ (HTML) documents 
    
    Returns: 
        faq_data[index][1] (str): Best response to the user-query               
    """    
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    best_match_index = similarities.argmax()
    
    if (not best_match_index):
        return None # if user query doesn't match any available content in FAQ documents
        
    return faq_data[best_match_index][1]

# main() Function
def main():
    print('\n NLP-based Query-Response Tool ')
    print(' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')
    folder_path = '.\\faq'                      # path to folder containing FAQ (HTML) documents
    faq_data = parse_html_files(folder_path)    # parse the HTML files
    vectorizer, tfidf_matrix = vectorize_text(faq_data)  # word-vectorisation of parsed text 
    
    while True:
        user_query = input("\n Enter your query: (type EXIT or QUIT to exit) \n \t")
        if user_query.lower() in ['exit', 'quit']:
            break
        
        response = find_best_match(user_query, vectorizer, tfidf_matrix, faq_data)
        if (not response): 
            print("\n \t Apologies! No suitable match found for your query!")
        else: 
            print(" Response to your query: \n \t", response)

if __name__ == "__main__":
    main()
