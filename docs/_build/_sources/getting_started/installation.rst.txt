Installation
============

Below is the command to install with pip. 

.. code-block:: bash

    pip install -U git+https://github.com/ComputerVisionLaboratory/cvlab_toolbox

We use a Scikit-learn API so it should be pretty easy to get your code up and running. Here's an example that should work copy&paste.

.. code-block:: python

    import numpy as np
    from numpy.random import randint, rand
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from cvt.models import KernelMSM

    dim = 100
    n_class = 4
    n_train, n_test = 20, 5

    # input data X is *list* of vector sets (list of 2d-arrays)
    X_train = [rand(randint(10, 20), dim) for i in range(n_train)]
    X_test = [rand(randint(10, 20), dim) for i in range(n_test)]

    # labels y is 1d-array
    y_train = randint(0, n_class, n_train)
    y_test = randint(0, n_class, n_test)

    model = KernelMSM(n_subdims=3, sigma=0.01)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(accuracy_score(pred, y_test))
