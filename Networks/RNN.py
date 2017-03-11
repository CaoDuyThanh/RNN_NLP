import pickle
import cPickle
from Layers.RNNHiddenLayer import *
from Utils.CostFHelper import *

class RNN:
    def __init__(self,
                 rng,
                 numIn,
                 numHidden,
                 truncate,
                 activation = T.tanh,
                 learningRate = 0.005):
        # Set parameters
        self.Rng = rng
        self.NumIn = numIn
        self.NumHidden = numHidden
        self.Truncate = truncate
        self.Activation = activation
        self.LearningRate = learningRate

        self.createRNN()

    def createRNN(self):
        # Save shared parameters
        self.Whh = None
        self.Wx = None
        self.Wy = None

        # Create RNN model
        self.Layers = []
        for layerId in range(self.Truncate):
            if layerId == 0:
                hiddenLayer = RNNHiddenLayer(
                    rng        = self.Rng,
                    numIn      = self.NumIn,
                    numHidden  = self.NumHidden,
                    sActivation = self.Activation
                )
                self.Whh = hiddenLayer.Whh
                self.Wx  = hiddenLayer.Wx
                self.Wy  = hiddenLayer.Wy
            else:
                if layerId == self.Truncate - 1:
                    hiddenLayer = RNNHiddenLayer(
                        rng         = self.Rng,
                        numIn       = self.NumIn,
                        numHidden   = self.NumHidden,
                        sActivation = self.Activation,
                        yActivation = T.nnet.softmax,
                        whh  = self.Whh,
                        wx   = self.Wx,
                        wy   = self.Wy
                    )
                else:
                    hiddenLayer = RNNHiddenLayer(
                        rng         = self.Rng,
                        numIn       = self.NumIn,
                        numHidden   = self.NumHidden,
                        sActivation = self.Activation,
                        whh  = self.Whh,
                        wx   = self.Wx,
                        wy   = self.Wy
                    )
            self.Layers.append(hiddenLayer)

        # Create train model
        X = T.ivector('X')
        Y = T.ivector('Y')

        # Feed-forward
        S = numpy.ones((1, self.NumHidden), dtype = theano.config.floatX)
        for idx, layer in enumerate(self.Layers):
            [S, Yp] = layer.FeedForward(S, X[idx])

        # Calculate cost | error function
        cost = CrossEntropy(Yp, Y)

        # Get params and calculate gradients
        params = self.Layers[-1].Params
        grads  = T.grad(cost, params)
        updates = [(param, param - self.LearningRate * grad)
                   for (param, grad) in zip(params, grads)]
        self.TrainFunc = theano.function(
            inputs  = [X, Y],
            outputs = cost,
            updates = updates,
        )

        # Create test model
        self.TestFunc = theano.function(
            inputs    = [X, Y],
            outputs   = cost
        )

    def LoadModel(self, fileName):
        file = open(fileName)
        self.Whh.set_value(cPickle.load(file), borrow = True)
        self.Wx.set_value(cPickle.load(file), borrow=True)
        self.Wy.set_value(cPickle.load(file), borrow=True)
        file.close()

    def SaveModel(self, fileName):
        file = open(fileName, 'wb')
        pickle.dump(self.Whh.get_value(borrow = True), file, -1)
        pickle.dump(self.Wx.get_value(borrow = True), file, -1)
        pickle.dump(self.Wy.get_value(borrow = True), file, -1)
        file.close()
