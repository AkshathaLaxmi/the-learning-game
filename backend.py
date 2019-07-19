
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from gensim.models import KeyedVectors
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import re
import sys

file_name = 'C:/Users/aksha/OneDrive/Desktop/Akshatha/moon.txt'
file = open(file_name, "r")
filedata = file.readlines()
    
def read_para(file_name, n):
    file = open(file_name, "r")
    filedata = file.readlines()
    para = filedata[n].split(". ")
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
def generate_summary(file_name, n, top_n=5):       
    stop_words = stopwords.words('english')
    summarize_text = []
    # Step 1 - Read text and tokenize
    sentences =  read_para(file_name, n)
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
pretrained_embeddings_path = 'C:/Users/aksha/Downloads/GoogleNews-vectors-negative300.bin'
word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)

def frame(summarize_text, key_words):
    count = 0
    l = ''
    summarize_text_lower = []
    key_words_lower = []
    for i in summarize_text:
        i = i.lower()
        a = i.split()
        for j in a:
                summarize_text_lower.append(j.replace("[^a-zA-Z]", " "))

    
    for j in key_words:
        key_words_lower.append(j.lower())
    
    for k in summarize_text_lower:
        if k in key_words_lower:
            if count==0:
                l = k
                count=+1
                print("_______", end = " ")        #change according to html
                
            else:
                print(k, end = " ")  
        else:
            print(k, end = " ")             #change according to html
    print()
    return l
file_out = open('C:/Users/aksha/OneDrive/Desktop/Akshatha/answer.txt', 'w+')
def check(in_word, actual):
    if in_word==actual:
        print("1", file = file_out)
        return 1
    else:
        print("0", file = file_out)
        return 0
def key_words(file_name, model):
    #if you want to use Google original vectors from Google News corpora
    #model = word2vec.Word2Vec.load_word2vec_format('C:/Users/aksha/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
    #if you want to use your own vector
    #model = word2vec.Word2Vec.load("w2v")
    file = open(file_name, "r")
    text_review = file.read()
    def text_to_wordlist(text, remove_stopwords=True):
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]", " ", text)

        # 3. Convert words to lower case and split them, clean stopwords from model' vocabulary
        words = review_text.lower().split()
        stops = set(stopwords.words('english'))
        meaningful_words = [w for w in words if not w in stops]
        return (meaningful_words)


    # Function to get feature vec of words
    def get_feature_vec(words, model):
        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed 
        index2word_set = set(model.index2word)
        clean_text = []
        # vocabulary, add its feature vector to the total
        for word in words:
            if word in index2word_set:
                clean_text.append(model[word])

        return clean_text


    # bag of word list without stopwords
    clean_train_text = (text_to_wordlist(text_review, remove_stopwords=True))

    # delete words which occur more than ones
    clean_train = []
    for words in clean_train_text:
        if words in clean_train:
            words = +1
        else:
            clean_train.append(words)

    trainDataVecs = get_feature_vec(clean_train, model)
    trainData = np.asarray(trainDataVecs)

    # calculate cosine similarity matrix to use in pagerank algorithm for dense matrix, it is not
    # fast for sparse matrix
    # sim_matrix = 1-pairwise_distances(trainData, metric="cosine")

    # similarity matrix, it is 30 times faster for sparse matrix
    # replace this with A.dot(A.T).todense() for sparse representation
    similarity = np.dot(trainData, trainData.T)

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag


    # pagerank powermethod
    def powerMethod(A, x0, m, iter):
        n = A.shape[1]
        delta = m * (np.array([1] * n, dtype='float64') / n)
        for i in range(iter):
            x0 = np.dot((1 - m), np.dot(A, x0)) + delta
        return x0


    n = cosine.shape[1]  # A is n x n
    m = 0.15
    x0 = [1] * n

    pagerank_values = powerMethod(cosine, x0, m, 130)

    srt = np.argsort(pagerank_values)
    a = srt[0:10]

    keywords_list = []

    for words in a:
        keywords_list.append(clean_train_text[words])
    print(keywords_list)
    return keywords_list
def answer():
    in_word = input("Ans: ")
    return in_word



points = 0
i = 0
while(i<len(filedata)):
    
    if len(filedata[i].split(". ")) >= 3:
        summarize_text = generate_summary(file_name, i, 2)

        keywords_list = key_words(file_name, word2vec)
        actual = frame(summarize_text, keywords_list)
        in_word = answer()
        n = check(in_word, actual)
        count = 0
#print('supermoon' in keywords_list)
    if n:
        if count==0:
            points+=1
            print("Correct. Points: ", points)
            count+=1
        
    else:
        print("Incorrect. Actual answer:", "\"", actual, "\"", "Points: ", points)
        print("Game Over")
        print("Click 1> To start over, 2> To start from the last question, 3> Exit:")
        m = int(input("Choice: "))
        if m == 1:
            i = 0
            points = 0
            continue
        elif m == 2:
            points = 0
            continue
        elif m==3:
            print("Thank")
            sys.exit()
        else:
            print("invalid input")
    i+=1  
#inp = answer()
#check(inp,summarize_text,n)
