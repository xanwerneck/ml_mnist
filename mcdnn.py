
import os
import sys
import timeit
import time
import cPickle

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from cnn import LeNetConvPoolLayer
import helper as helper


class DNNColumn(object):

    def __init__(self, ds=None, nkerns=[32, 48], batch_size=100, normalized_width=0, distortion=0,
                    params=[None, None,None, None,None, None,None, None]):

        #layers
        layer3_W, layer3_b, layer2_W, layer2_b, layer1_W, layer1_b, layer0_W, layer0_b = params        
        rng = numpy.random.RandomState(23455)

        #dataset for train
        train_set_x, train_set_y = ds[0]
        valid_set_x, valid_set_y = ds[1]
        test_set_x, test_set_y   = ds[2]

        # compute number of minibatches for training, validation and testing
        self.n_train_batches  = train_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches  = valid_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches   = test_set_x.get_value(borrow=True).shape[0]
        
        self.n_train_batches /= batch_size
        self.n_valid_batches /= batch_size
        self.n_test_batches  /= batch_size


        index = T.lscalar()
        learning_rate = T.fscalar()

        # start-snippet-1
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################

        print '... building the dnn column'

        layer0_input = x.reshape((batch_size, 1, 29, 29))

        # Construct the first convolutional pooling layer
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, 29, 29),
            filter_shape=(nkerns[0], 1, 4, 4),
            poolsize=(2, 2),
            W=layer0_W,
            b=layer0_b
        )

        # Construct the second convolutional pooling layer
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 13, 13),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(3, 3),
            W=layer1_W,
            b=layer1_b
        )

        layer2_input = layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * 3 * 3,
            n_out=150,
            W=layer2_W,
            b=layer2_b,
            activation=T.tanh
        )
        
        layer3 = LogisticRegression(
            input=layer2.output, 
            n_in=150, 
            n_out=10,
            W=layer3_W,
            b=layer3_b
        )

        cost = layer3.negative_log_likelihood(y)

        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # create a function to compute probabilities of all output classes
        self.test_output_batch = theano.function(
            [index],
            layer3.p_y_given_x,
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.validate_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.params = layer3.params + layer2.params + layer1.params + layer0.params
        self.column_params = [nkerns, batch_size, normalized_width, distortion]

        grads  = T.grad(cost, self.params)

        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, grads)
        ]

        self.train_model = theano.function(
            [index, learning_rate],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
    
    def test_outputs(self):
        test_losses = [
            self.test_output_batch(i)
            for i in xrange(self.n_test_batches)
        ]
        return numpy.concatenate(test_losses)

    def train_column(self, n_epochs=800,init_learning_rate=0.001):
        ######################
        # TRAIN MODEL COLUMN #
        ######################
        print '... training'
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(self.n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False
        
        while (epoch < n_epochs) and (not done_looping):
            current_learning_rate = max(numpy.array([init_learning_rate * 0.993**epoch, 0.00003], dtype=numpy.float32))
            epoch = epoch + 1

            for minibatch_index in xrange(self.n_train_batches):

                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print 'training @ iter = ', iter
                
                cost_ij = self.train_model(minibatch_index, current_learning_rate)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in xrange(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            self.test_model(i)
                            for i in xrange(self.n_test_batches)
                        ]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    def save(self, filename=None):
        """
        Will need to load last layer W,b to first layer W,b
        """
        name = filename or 'CNN_%iLayers_t%i' % (len(self.params) / 2, int(time.time()))

        print('Saving Model as "%s"...' % name)
        f = open('./models/'+name+'.pkl', 'wb')

        cPickle.dump([param.get_value(borrow=True) for param in self.params], f, -1)
        cPickle.dump(self.column_params, f, -1)
        f.close()
 

def train_mcdnn_column(normalized_width=0, n_epochs=800, trail=0):
    print '... train %i column of normalization %i' % (trail, normalized_width)
    print '... num_epochs %i' % (n_epochs)
    datasets = load_data(dataset='mnist.pkl.gz', digit_normalized_width=normalized_width, digit_out_image_size=29)
    column = DNNColumn(ds=datasets, normalized_width=normalized_width)
    column.train_column(n_epochs=n_epochs, init_learning_rate=0.1)
    filename = 'mcdnn_nm%i_trail%i_Layers_time_%i' % (normalized_width, trail, int(time.time()))
    column.save(filename)

if __name__ == '__main__':
    for nm in [0,10,12,14,16,18,20]:
        for trail in [0,1,2,3,4]:
            train_mcdnn_column(nm, n_epochs=800, trail=trail)
