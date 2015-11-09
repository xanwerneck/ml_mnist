
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from cnn import LeNetConvPoolLayer
import helper as helper

def inspect_inputs(i, node, fn):
	print i, node, "input(s) value(s):", [input[0] for input in fn.inputs]

def inspect_outputs(i, node, fn):
	print "output(s) value(s):", [output[0] for output in fn.outputs]


class DeepConvolutionalNeuralNetwork(object):

    def __init__(self,batch_size=500,x=None,nkerns=[20,40]):

	    layer0_input = x.reshape((batch_size, 1, 28, 28))
	    rng = numpy.random.RandomState(23455)

	    # Construct the first convolutional pooling layer
	    layer0 = LeNetConvPoolLayer(
	        rng,
	        input=layer0_input,
	        image_shape=(batch_size, 1, 28, 28),
	        filter_shape=(nkerns[0], 1, 4, 4),
	        poolsize=(2, 2)
	    )

	    # Construct the second convolutional pooling layer
	    layer1 = LeNetConvPoolLayer(
	        rng,
	        input=layer0.output,
	        image_shape=(batch_size, nkerns[0], 12, 12),
	        filter_shape=(nkerns[1], nkerns[0], 5, 5),
	        poolsize=(3, 3)
	    )

	    layer2_input = layer1.output.flatten(2)

	    # construct a fully-connected sigmoidal layer
	    layer2 = HiddenLayer(
	        rng,
	        input=layer2_input,
	        n_in=nkerns[1] * 3 * 3,
	        n_out=150,
	        activation=T.tanh
	    )
	    
	    # classify the values of the fully-connected sigmoidal layer
	    self.layer3 = LogisticRegression(input=layer2.output, n_in=150, n_out=10)


def mcddn(learning_rate=0.001, n_epochs=800,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 40], batch_size=500):

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    #daaset with additional normalized dataset
    train_set_x, train_set_y = datasets[0]
    
    #normalize digit width
    #train_set_x_norm, train_set_y_norm = helper.additional_database(train_set_x,train_set_y)    
    
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y   = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches  = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches  = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches   = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches  /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    dnn = DeepConvolutionalNeuralNetwork(
        batch_size=batch_size,
        x=x,
        nkerns=nkerns
    )

    cost = dnn.layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        dnn.layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        dnn.layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    params = dnn.layer3.params
    grads  = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
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
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
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
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
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

if __name__ == '__main__':
    mcddn()
