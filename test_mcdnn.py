import sys

import cPickle
import numpy
import theano
import theano.tensor as T

from logistic_sgd import load_data
from mcdnn import DNNColumn

import pdb

def test_columns(exclude_mode, models, valid_test='V'):
    dataset='mnist.pkl.gz'
    print '... Starting to test %i columns' % len(models)
    # create data hash that will be filled with data from different normalizations
    all_datasets = {}
    # instantiate multiple columns
    columns = []
    for model in models:
        # load model params
        f = open('./models/'+model)
        params = cPickle.load(f)
        nkerns, batch_size, normalized_width, distortion = cPickle.load(f)
        if all_datasets.get(normalized_width):
            datasets = all_datasets[normalized_width]
        else:
            datasets = load_data(dataset, normalized_width, 29)
            all_datasets[normalized_width] = datasets
        # no distortion during testing
        columns.append(DNNColumn(datasets, nkerns, batch_size, normalized_width, 0, params))
    print '... Forward propagating %i columns' % len(models)
    # call test on all of them recieving 10 outputs
    if valid_test=='V':
        model_outputs = [column.valid_outputs() for column in columns] 
        position_ds   = 1 
    else:
        model_outputs = [column.test_outputs() for column in columns]      
        position_ds   = 2
    # average 10 outputs
    avg_output = numpy.mean(model_outputs, axis=0)
    # argmax over them
    predictions = numpy.argmax(avg_output, axis=1)
    # compare predictions with true labels
    pred = T.ivector('pred')

    all_true_labels_length = theano.function([], all_datasets.values()[0][position_ds][1].shape)
    remainder = all_true_labels_length() - len(predictions)
    if exclude_mode and remainder:
        print '... Excluding FIRST %i points' % remainder
        true_labels = all_datasets.values()[0][position_ds][1][remainder:]
    elif remainder: # TODO: remove this, doesn't seem to make sense since the predictions would be misaligned
        print '... Excluding LAST %i points' % remainder
        true_labels = all_datasets.values()[0][position_ds][1][:len(predictions)]
    else:
        true_labels = all_datasets.values()[0][position_ds][1][:]

    error = theano.function([pred], T.mean(T.neq(pred, true_labels)))
    acc = error(predictions.astype(dtype=numpy.int32))
    print '....'
    print 'Error across %i columns: %f %%' % (len(models), 100*acc)
    return [predictions, acc]

if __name__ == '__main__':
    # how to test
    # ex.: python test_mcdnn.py 0 DNN_4Layers_t1448204295.pkl
    assert len(sys.argv) > 2
    valid_test = sys.argv[2]
    if valid_test == 'V':
        print '... executing validation on models'
        models = sys.argv[3:]
        test_columns(int(sys.argv[1]), models, 'V')
    else:
        print '... executing test on models'
        models = sys.argv[3:]
        test_columns(int(sys.argv[1]), models, 'T')