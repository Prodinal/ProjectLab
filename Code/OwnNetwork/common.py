
import numpy

__all__ = (
    'OUTCOMES',
    'sigmoid',
    'softmax',
	'TEST_NUM'
)

OUTCOMES = ["Malignant", "Benign"]

TEST_NUM = 20

def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:,numpy.newaxis]

def sigmoid(a):
    return 1. / (1. + numpy.exp(-a))