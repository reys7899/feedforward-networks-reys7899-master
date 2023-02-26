# Objectives

The learning objectives of this assignment are to:
1. get familiar with the TensorFlow Keras framework for training neural networks.
2. experiment with the various hyper-parameter choices of feedforward networks.
We will implement several feedforward neural networks using the TensorFlow Keras API.

# Environment Setup

* [Python (version 3.8 or higher)](https://www.python.org/downloads/)
* [tensorflow (version 2.9)](https://www.tensorflow.org/)
* [pytest](https://docs.pytest.org/)

# Tests

Tests have been provided for you in the `test_nn.py` file.
The tests show how each of the methods is expected to be used.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
...
collected 4 items

test_nn.py FFFF                                                          [100%]

=================================== FAILURES ===================================
...
============================== 4 failed in 7.33s ===============================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
...
collected 4 items

test_nn.py
8.2 RMSE for baseline on Auto MPG
6.2 RMSE for deep on Auto MPG
3.9 RMSE for wide on Auto MPG
.
65.0% accuracy for baseline on del.icio.us
68.7% accuracy for relu on del.icio.us
66.9% accuracy for tanh on del.icio.us
.
18.2% accuracy for baseline on UCI-HAR
93.8% accuracy for dropout on UCI-HAR
91.7% accuracy for no dropout on UCI-HAR
.
75.4% accuracy for baseline on census income
79.0% accuracy for early on census income
77.8% accuracy for late on census income
.                                                          [100%]

============================== 4 passed in 23.16s ==============================
```
**Warning**: The performance of your models may change somewhat from run to run,
especially when moving from one machine to another, since neural network models
are randomly initialized.

# Acknowledgments

The author of the test suite (test_nn.py) is Dr. Steven Bethard.
