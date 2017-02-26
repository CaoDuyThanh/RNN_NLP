from Utils.DataHelper import *

DATASET_NAME = '../Dataset/reddit-comments-2015-08.csv'
SAVE_PATH = '../Pretrained/model.pkl'

# Hyper parameters
VOCABULARY_SIZE = 8000
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
LEARNING_RATE = 0.005
NUM_EPOCH = 1500
BATCH_SIZE = 50
PATIENCE = 1000
PATIENCE_INCREASE = 2
IMPROVEMENT_THRESHOLD = 0.995
VALIDATION_FREQUENCY = 1000
VISUALIZE_FREQUENCY = 5000



def NLP():
    # Load datasets from local disk reddit-comments-2015-08.csv
    sentences = LoadData(DATASET_NAME)

    #############################
    #    PRE-PROCESSING DATA    #
    #############################
    # Tokenize the sentences into words
    tokenized = [nltk.word_tokenize(sent) for sent in sentences]
    # Count the word frequency
    wordFreq = nltk.FreqDist(itertools.chain(*tokenized))

    # Get the most frequency words and build indexToWord and wordToIndex
    vocab = wordFreq.most_common(VOCABULARY_SIZE - 1)
    indexToWord = [x[0] for x in vocab]
    indexToWord.append(UNKNOWN_TOKEN)
    wordToIndex = dict([(w, i) for i, w in enumerate(indexToWord)])

    print ('Vocabulary size ', VOCABULARY_SIZE)
    print ('The least frequent word in our vocabulary is ', vocab[-1][0], ' and appear ', vocab[-1][1],' times')

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized):
        tokenized[i] = [w if w in wordToIndex else UNKNOWN_TOKEN]

    print ('Example sentence: ', sentences[0])
    print ('Example sentence after preprocessing: ', tokenized[0])

    #############################
    #        BUILD MODEL        #
    #############################



    return 0

if __name__ == '__main__':
    NLP()