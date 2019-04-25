import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
import gensim


#from nltk.stem.porter import PorterStemmer
dataset = pd.read_excel(r'C:\Users\varsha.vishwakarma\Documents\Document-classification\data.xlsx')
dataset = dataset.drop("Cluster", axis=1)
dataset = dataset.drop("JD_Name", axis=1)
df_p = dataset.loc[(dataset['Similar'] == 1)]
df_p = df_p.drop("Similar", axis=1)
df_n = dataset.loc[(dataset['Similar'] == 0)]
df_n = df_n.drop("Similar", axis=1)
jd_des = df_p["JD_Description"].tolist()
jd_exp = df_p["JD_Experience"].tolist()
jd_edu = df_p["JD_Education"].tolist()
cv_sum = df_p["CV_Summary"].tolist()
cv_exp = df_p["CV_Experience"].tolist()
cv_ski = df_p["CV_Skill"].tolist()
cv_edu = df_p["CV_Education"].tolist()

ls = jd_des + jd_exp + jd_edu + cv_sum + cv_exp + cv_ski +cv_edu 

stopwords=stopwords.words('english')
stopwords.append(".")
ps = PorterStemmer()
new_wv=[]
regex = re.compile('[^a-zA-Z.]')
for i in ls:
    temp=regex.sub(' ', i)
    tokens = nltk.word_tokenize(temp)
    tagged = nltk.pos_tag(tokens)
    tg=[t[0] for t in tagged if t[1].startswith('N') ]
    token = [k.lower() for k in tokens if k.lower() not in stopwords]        #token = [ps.stem(k) for k in token]
    new_wv.append(token)
#new_wv = list(set(new_wv))
        
modeled = gensim.models.Word2Vec(new_wv,size=100,window=5,min_count=1, workers=2, sg=1)
modeled.train(new_wv, total_examples=len(new_wv), epochs=500)
modeled.save(r"C:\Users\varsha.vishwakarma\Documents\Document-classification\model\word2vec.model")



X_train_p, X_test = train_test_split( df_p, test_size=0.2, random_state=42)
#df_p.to_csv(r'C:\Users\varsha.vishwakarma\Documents\CNN-text-classification\out_positive.csv', header=False,index=False)
#df_n.to_csv(r'C:\Users\varsha.vishwakarma\Documents\CNN-text-classification\out_negative.csv', header=False,index=False)

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    #positive_examples = []
    positive_examples = list(open(r'C:\Users\varsha.vishwakarma\Documents\CNN-text-classification\work-day-classification\out_positive.csv', "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(r'C:\Users\varsha.vishwakarma\Documents\CNN-text-classification\work-day-classification\out_negative.csv', "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    
    x_text = positive_examples + negative_examples
    #x_text = negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    #y = np.concatenate([negative_labels], 0)
    y = np.concatenate([positive_labels, negative_labels], 0)
    
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    data = []
    #text = parser.from_file(filename)
    #content = dataset['content']
    for i in range(len(sentences)):
         content_cleaned = " ".join(sentences[i])
         #print(content_cleaned)
         #content_cleaned = sentences[i]
         content_cleaned = re.sub(r'[^\w]', ' ', content_cleaned)
         content_cleaned = re.sub("^\b\d+\b", " ", content_cleaned)
         content_cleaned = re.sub(r'[^a-zA-Z ]', '', content_cleaned)
         english_stopwords = stopwords.words('english')
         content_cleaned=[word for word in content_cleaned.split() if word.lower() not in english_stopwords]
         #content_cleaned= " ".join(content_cleaned)
         data.append(content_cleaned)
    

    sequence_length = max(len(x) for x in data)
    #print(sequence_length)
    padded_sentences = []
    for i in range(len(data)):
        sentence = data[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = list(set(sorted(word_counts)))
    
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    
    sentences_padded = pad_sentences(sentences)
    #return sentences_padded
    #print()
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, sentences_padded]

#x, y, vocabulary, vocabulary_inv, sentences_padded = load_data()
#x = x[140:146,:]
#df_x = pd.DataFrame(x)
#y = y[140:146,:]
#df_y = pd.DataFrame(y)
#df_x.to_csv(r'C:\Users\varsha.vishwakarma\Documents\CNN-text-classification\test_input_dataset_withjd.csv', header=False,index=False)
#df_y.to_csv(r'C:\Users\varsha.vishwakarma\Documents\CNN-text-classification\test_output_dataset_withjd.csv', header=False,index=False)