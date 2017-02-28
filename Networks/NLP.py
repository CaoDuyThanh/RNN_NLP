import os.path
from Utils.DataHelper import *
from RNN import *

DATASET_NAME = '../Data/reddit-comments-2015-08.csv'
# DATASET_NAME = '../Data/minidata.csv'
SAVE_PATH = '../Pretrained/model.pkl'

# Hyper parameters
VOCABULARY_SIZE = 8000
NUM_HIDDEN = 80
TRUNCATE = 4
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
LEARNING_RATE = 0.005
NUM_EPOCH = 200
PATIENCE = 1000
PATIENCE_INCREASE = 2
IMPROVEMENT_THRESHOLD = 0.995
VALIDATION_FREQUENCY = 1000

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
        tokenized[i] = [w if w in wordToIndex else UNKNOWN_TOKEN for w in sent]

    print ('Example sentence: ', sentences[0])
    print ('Example sentence after preprocessing: ', tokenized[0])

    # Create training dataset
    trainDatasetX = numpy.asarray([[wordToIndex[w] for w in sent[:-1]] for sent in tokenized])
    trainDatasetY = numpy.asarray([[wordToIndex[w] for w in sent[1:]] for sent in tokenized])

    #############################
    #        BUILD MODEL        #
    #############################
    rng = numpy.random.RandomState(123)
    rnnModel = RNN(
        rng = rng,
        numIn = VOCABULARY_SIZE,
        numHidden = NUM_HIDDEN,
        truncate = TRUNCATE,
        useSoftmax = True,
    )

    # Train model - using early stopping
    # Load old model if exist
    if os.path.isfile(SAVE_PATH):
        print ('Load old model and continue the training')
        rnnModel.LoadModel(SAVE_PATH)

    # Gradient descent - early stopping
    epoch = 0
    iter = 0
    patient = PATIENCE
    doneLooping = False
    bestCost = 10000
    while (epoch < NUM_EPOCH) and (not doneLooping):
        epoch += 1

        # Travel through the training set
        for sent in trainDatasetX:
            iter += 1

            # Calculate cost of trainModel
            for idx in range(len(sent) - TRUNCATE):
                XTrain = numpy.zeros((TRUNCATE, VOCABULARY_SIZE), dtype = theano.config.floatX)
                XTrain[range(TRUNCATE), sent[idx:idx + TRUNCATE]] = 1
                YTrain = numpy.zeros((1, VOCABULARY_SIZE), dtype = theano.config.floatX)
                YTrain[0, sent[idx + TRUNCATE]] = 1
                cost = rnnModel.TrainModel(XTrain, YTrain)

            print cost

            # Calculate cost of validation set every VALIDATION_FREQUENCY iter
            # if iter % VALIDATION_FREQUENCY == 0:
            #     costValid = validModel(ind)
            #
            #     if costValid < bestCost:
            #         bestCost = costValid
            #         rnnModel.SaveModel(SAVE_PATH)

        if patient < iter:
            doneLooping = True
            break

    # Load model and test
    if os.path.isfile(SAVE_PATH):
        print ('Can not find old model !')
        rnnModel.LoadModel(SAVE_PATH)


    # print ('Cost of test model : ', costTest)

if __name__ == '__main__':
    NLP()