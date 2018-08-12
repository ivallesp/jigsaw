import unittest
import math
import numpy as np
import tensorflow as tf
from src.data_tools import load_train_data

class DataTester(unittest.TestCase):
    def test_data_reproducibility(self):
        df_train_1, df_dev_1, df_test_1, _, _ = load_train_data()
        df_train_2, df_dev_2, df_test_2, _, _ = load_train_data()
        self.assertTrue(df_train_1.equals(df_train_2))
        self.assertTrue(df_dev_1.equals(df_dev_2))
        self.assertTrue(df_test_1.equals(df_test_2))
