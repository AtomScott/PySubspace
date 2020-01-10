.. cvlab_toolbox documentation master file, created by
   sphinx-quickstart on Thu Dec 12 20:08:18 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cvlab_toolbox's documentation!
=========================================

.. Warning::
    This is not yet released. Privacy levels are set to `Private <https://docs.readthedocs.io/en/stable/privacy.html>`_ and therefore this should only be viewable by direct search from a link.

    Below are details on the privacy levels.

    +------------+------------+-----------+-----------+-------------+
    | Level      | Detail     | Listing   | Search    | Viewing     |
    +============+============+===========+===========+=============+
    | Private    | No         | No        | No        | Yes         |
    +------------+------------+-----------+-----------+-------------+
    | Public     | Yes        | Yes       | Yes       | Yes         |
    +------------+------------+-----------+-----------+-------------+

This is the repository of CVLAB toolbox, which contains various subspace methods for classification.

All of the code is from the Computer Vision Laboratory (CVLAB), Graduate school of Systems and Information Engineering, University of Tsukuba (`web <https://en.home.cvlab.cs.tsukuba.ac.jp/>`_). Please check the `github repo <https://github.com/ComputerVisionLaboratory/cvlab_toolbox>`_ for individual credits.

- We are always looking for motivated students to help us with our efforts. If you would like to join our lab, please contact with Prof. Fukui via e-mail.
- Our laboratory serves as one of the machine learning units in `the Center for Artificial Intelligence Research (C-AIR) <https://air.tsukuba.ac.jp/en/faculty/>`_.
- Our laboratory also is involved with `the Empowerment Informatics Program for Leading Graduate Schools <http://www.emp.tsukuba.ac.jp/english/>`_. If you are interested in this program, please feel free to contact Prof. Fukui or Prof. Iizuka with any questions.




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


.. toctree:: 
   :maxdepth: 1
   :caption: Getting Started

   ./getting_started/installation

.. toctree:: 
   :maxdepth: 1
   :caption: Tutorials

   ./tutorials/getting_started
   ./tutorials/index
   ./tutorials/references

.. toctree::
   :maxdepth: 2
   :caption: Examples

   ./examples/MNIST_example_with_SM
   ./examples/pca_to_sm


.. toctree::
    :maxdepth: 4
    :caption: API reference
    
    ./source/cvt

.. toctree::
    :maxdepth: 2
    :caption: Contribution

    ./contribution/contribution

.. toctree::
    :maxdepth: 2
    :caption: Gallery

    ./examples_scripts/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
