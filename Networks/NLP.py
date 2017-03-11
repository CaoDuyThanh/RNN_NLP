import os.path
from Utils.DataHelper import *
from RNN import *

DATASET_NAME = '../Data/reddit-comments-2015-08.csv'
# DATASET_NAME = '../Data/minidata.csv'
SAVE_PATH = '../Pretrained/model.pkl'

# TRAINING PARAMETERS
VOCABULARY_SIZE = 8000
NUM_HIDDEN = 80
TRUNCATE = 4
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
LEARNING_RATE = 0.005
NUM_EPOCH = 200
PATIENCE = 1000
PATIENCE_INCREASE = 2
IMPROVEMENT_THRESHOLD = 0.995

VISUALIZE_FREQUENCY = 1000
VALIDATION_FREQUENCY = 300000

PERCENT_TRAIN = 0.80
PERCENT_VALID = 0.05
PERCENT_TEST  = 0.15


def testModel(rnnModel, validDatasetX):
    print ('Validate model...')
    iter = 0
    cost = 0
    for sent in validDatasetX:
        # Calculate cost of trainModel
        for idx in range(len(sent) - TRUNCATE):
            iter += 1
            XTrain = sent[idx : idx + TRUNCATE]
            YTrain = sent[idx + TRUNCATE : idx + TRUNCATE + 1]
            cost += rnnModel.TestFunc(XTrain, YTrain)

            if (iter % VISUALIZE_FREQUENCY == 0):
                print ('     Iteration =  %d, cost = %f ' % (iter, cost))
    return cost / iter

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
    numSamples = len(tokenized)
    sliceIndex = numpy.array([0, numSamples * PERCENT_TRAIN, numSamples * (PERCENT_TRAIN + PERCENT_VALID), numSamples], dtype = 'int32')
    trainDatasetX = numpy.asarray([[wordToIndex[w] for w in sent[:-1]] for sent in tokenized[sliceIndex[0]:sliceIndex[1]]])
    validDatasetX = numpy.asarray([[wordToIndex[w] for w in sent[1:]] for sent in tokenized[sliceIndex[1]:sliceIndex[2]]])
    testDatasetX = numpy.asarray([[wordToIndex[w] for w in sent[1:]] for sent in tokenized[sliceIndex[2]:sliceIndex[3]]])

    #############################
    #        BUILD MODEL        #
    #############################
    rng = numpy.random.RandomState(123)
    rnnModel = RNN(
        rng       = rng,
        numIn     = VOCABULARY_SIZE,
        numHidden = NUM_HIDDEN,
        truncate  = TRUNCATE
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
    trainCost = []
    while (epoch < NUM_EPOCH) and (not doneLooping):
        epoch += 1

        # Travel through the training set
        for sent in trainDatasetX:
            # Calculate cost of trainModel
            for idx in range(len(sent) - TRUNCATE):
                iter += 1
                XTrain = sent[idx : idx + TRUNCATE]
                YTrain = sent[idx + TRUNCATE : idx + TRUNCATE + 1]
                cost = rnnModel.TrainFunc(XTrain, YTrain)
                trainCost.append(cost)

                # Calculate cost of validation set every VALIDATION_FREQUENCY iter
                if iter % VALIDATION_FREQUENCY == 0:
                    # Calculate cost of validModel
                    costValid = testModel(rnnModel, validDatasetX)

                    if costValid < bestCost:
                        print ('Save model ! Valid cost = ', costValid)
                        bestCost = costValid
                        rnnModel.SaveModel(SAVE_PATH)

                if (iter % VISUALIZE_FREQUENCY == 0):
                    print ('Epoch = %d, iteration =  %d, cost = %f ' % (epoch, iter, numpy.mean(trainCost)))
                    trainCost = []


    # Load model and test
    if os.path.isfile(SAVE_PATH):
        rnnModel.LoadModel(SAVE_PATH)
        costTest = testModel(rnnModel, testDatasetX)
        print ('Cost of test dataset = ', costTest)
    else:
        print ('Can not find old model !')

    # print ('Cost of test model : ', costTest)

if __name__ == '__main__':
    NLP()