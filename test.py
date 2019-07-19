#!C:/Python34/python
print("Content-Type: text/html")
print()

import cgi
import PyPDF2
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from gensim.models import KeyedVectors
import gensim
from sklearn.metrics.pairwise import cosine_similarity
#def read_file():

    
def read_para(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    para = filedata[0].split(". ")
    sentences = []
    for sentence in para:
     #print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
#extraction summarization
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix
def generate_summary(file_name, top_n=5):       
    stop_words = stopwords.words('english')
    summarize_text = []
    # Step 1 - Read text and tokenize
    sentences =  read_para(file_name)
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    #save the summary
    #print("Summarize Text: \n", ". ".join(summarize_text))
    return summarize_text
#pretrained_embeddings_path = 'C:/Users/aksha/Downloads/GoogleNews-vectors-negative300.bin'
#word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)
def frame(summarize_text):                  
    for i in summarize_text:
        score.append(model.simiarity(i, summarize_text))
    n = index(max(score))
    for i in summarize_text:
        if i==summarize_text[n]:
            print("<p>___________</p>", end = "<p> </p>")            #change according to html
        else:
            print(i, end = "<p> </p>")             #change according to html
    print()
    return n
def answer():
    form=cgi.FieldStorage()
    inp=form."inp"
    return inp

def check(in_word, summarize_text, n):
    if in_word==summarize_text[n]:
        return 1
    else:
        return 0
file_name = 'C:/Users/aksha/OneDrive/Desktop/Akshatha/moon.txt'
summarize_text = generate_summary(file_name, 2)
n = frame(summarize_text)
inp = answer()
check(inp,summarize_text,n)
