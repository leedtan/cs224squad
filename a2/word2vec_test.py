import unittest
import random
import numpy as np
import dataclasses

from utils import utils 
from utils import gradcheck as gradcheck_utils
import word2vec


@dataclasses.dataclass
class Word2VecState:
    center

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        def dummySampleTokenIdx():
            return random.randint(0, 4)

        def getRandomContext(C):
            tokens = ["a", "b", "c", "d", "e"]
            return tokens[random.randint(0,4)], \
                [tokens[random.randint(0,4)] for i in range(2*C)]

        dataset = type('dummy', (), {})()
        dataset.sampleTokenIdx = dummySampleTokenIdx
        dataset.getRandomContext = getRandomContext

        random.seed(31415)
        np.random.seed(9265)
        dummy_vectors = utils.normalizeRows(np.random.randn(10,3))
        dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
        
        self.dataset = dataset
        self.dummy_vectors = dummy_vectors
        self.dummy_tokens = dummy_tokens

    def test_naiveSoftmaxLossAndGradient(self):
        """ Test naiveSoftmaxLossAndGradient """

        centerWordVec = np.array([0 , 1])

        outsideWordVectors = np.array([
            [0., 0.],
            [0., 1.],
            [1., 1.],
            [1., 0.],
        ])


        loss, gradCenterVec, gradOutsideVecs = word2vec.naiveSoftmaxLossAndGradient(
            centerWordVec=centerWordVec, outsideWordIdx=0, outsideVectors=outsideWordVectors, dataset=None)
        print(loss)
        print(gradCenterVec)
        print(gradOutsideVecs)
        loss, gradCenterVec, gradOutsideVecs = word2vec.naiveSoftmaxLossAndGradient(
            centerWordVec=centerWordVec, outsideWordIdx=1, outsideVectors=outsideWordVectors, dataset=None)
        print(loss)
        print(gradCenterVec)
        print(gradOutsideVecs)
        loss, gradCenterVec, gradOutsideVecs = word2vec.naiveSoftmaxLossAndGradient(
            centerWordVec=centerWordVec, outsideWordIdx=2, outsideVectors=outsideWordVectors, dataset=None)
        print(loss)
        print(gradCenterVec)
        print(gradOutsideVecs)
        loss, gradCenterVec, gradOutsideVecs = word2vec.naiveSoftmaxLossAndGradient(
            centerWordVec=centerWordVec, outsideWordIdx=3, outsideVectors=outsideWordVectors, dataset=None)
        print(loss)
        print(gradCenterVec)
        print(gradOutsideVecs)

if __name__ == '__main__':
    unittest.main()