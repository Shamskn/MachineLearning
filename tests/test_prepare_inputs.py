import unittest
import numpy as np
from utils import prepare_inputs

class TestPrepareInputs(unittest.TestCase):

    def test_array_dim(self):
        self.assertRaises(ValueError, prepare_inputs.convert_array, [1, 2, 3, 4])
        self.assertRaises(ValueError, prepare_inputs.check_X_y, [1, 2, 3, 4], [1, [2]])

    def test_array_type(self):
        X = prepare_inputs.convert_array([[1,2],[3,4]])
        self.assertIsInstance(X, np.ndarray)
        self.assertTrue(X.dtype == np.float64)

    def test_equal_lengths(self):
        self.assertRaises(ValueError, prepare_inputs.equal_lengths, [2, 3, 4], [3, 3])

    def test_target_is_binary(self):
        self.assertFalse(prepare_inputs.target_is_binary(np.array([1, 0, 1, 3])))
        self.assertTrue(prepare_inputs.target_is_binary(np.array([1, 0, 1, 0])))

if __name__ == '__main__':
    unittest.main()


