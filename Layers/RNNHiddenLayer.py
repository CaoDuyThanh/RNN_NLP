import theano
import theano.tensor as T
import numpy

class RNNHiddenLayer:


    def __init__(self,
                 rng,
                 numIn,
                 numHidden,
                 whh = None,
                 wx = None,
                 wy = None,
                 activation = T.tanh):
        # Set parameters
        self.Rng = rng
        self.NumIn = numIn
        self.NumHidden = numHidden
        self.Whh = whh
        self.Wx = wx
        self.Wy = wy
        self.Activation = activation

        self.createHiddenLayer()
        self.buildHiddenLayer()

    def createHiddenLayer(self):
        if self.Whh is None:
            self.Whh = theano.shared(
                numpy.asarray(self.Rng.uniform(
                        low = -1.0,
                        high = 1.0,
                        size = (self.NumHidden, self.NumHidden)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        if self.Wx is None:
            self.Wx = theano.shared(
                numpy.asarray(self.Rng.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (self.NumIn, self.NumHidden)
                ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        if self.Wy is None:
            self.Wy = theano.shared(
                numpy.asarray(self.Rng.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (self.NumHidden, self.NumIn)
                ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

    def buildHiddenLayer(self):
        self.Params = [self.Whh, self.Wx, self.Wy]

    def FeedForward(self, Skm1, Xk):
        S = T.dot(Xk, self.Wx) + T.dot(Skm1, self.Whh)
        Y = T.dot(S, self.Wy)
        if self.Activation is None:
            return [S, Y]
        else:
            return [self.Activation(S), self.Activation(Y)]
