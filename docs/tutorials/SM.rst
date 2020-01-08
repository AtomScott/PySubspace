Subspace Method
===============

Theory
------

The Subspace method assumes an input vector :math:`\mathbf{x}'` and :math:`k`-class subspaces. Each class subspace approximates a data distribution for a single class. This approximation is obtained by applying `PCA <PCA.html>`_ to each class :cite:`Fukui2014`.

.. note::

   Check out the tutorial, "`from pca to the subspace method <../examples/pca_to_sm.html>`_" for a detailed walkthrough for obtaining subspaces using pca.

.. figure:: ../_static/tutorials/SM.png
    :align: center

    Fig.1

In Detail
---------

The similarity :math:`S` of the input vector :math:`\mathbf{x}'` to the :math:`i^{th}` class subspace :math:`Q_i` is defined based on either:

* The length of the  projection of :math:`\mathbf{x}'` to :math:`Q_i` :cite:`watanabe1967evaluation`.
* The minimum angle between :math:`\mathbf{x}'` and :math:`Q_i` :cite:`iijima1974theory`.

The length of an input vector :math:`\mathbf{x}'` is often normalized to 1.0. In this case these two criteria are identical. This fact should be obvious from Fig.1.

Since they are the same, from here on we think of the angle-based similarity :math:`S` defined by the following equation:

.. math::
    S = \cos{\theta} = \sum^{k}_{i=1} \frac{(\mathbf{x}' · \Phi_i)^2}{||\mathbf{x}'||^2}

:math:`\Phi_i` is the :math:`i^{th}` orthogonal normal basis vector of the class subspace :math:`Q_i`, which are obtained from applying the principal component analysis (PCA) to a set of patterns of the class. In more rigourus terms, these orthonormal basis vectors can be obtained as the eigenvectors of the correlation matrix 

.. math::
    \sum^{l}_{i=1}\mathbf{x}^{(i)}\mathbf{x}^{(i)\top} where \mathbf{x}^{(i)} \in \mathbb{X} 
    
of the class (:math:`\mathbb{X}` is the training dataset).

Learning Phase
~~~~~~~~~~~~~~

1. Generate :math:`k` class subspaces from each class by using PCA.

Recognition Phase
~~~~~~~~~~~~~~~~~

1. Calculate :math:`S` between :math:`\mathbf{x}` and each subspace :math:`Q_i`. 
2. Classify the :math:`\mathbf{x}` into the class where :math:`S` was calculated to be the highest.

Also, you may add a rejection thershold on :math:`S` to reject classifications with low similarity.