Principle Component Analysis
============================

PCA aims to find linearly uncorrelated orthogonal axes, which are also known as principal components (PCs)in the m dimensional space to project the data points onto. The first PC captures the largest variance in the data and the second PC captures the second largest variance a direction which is orthogonal to the first. 

Below is 2-D scatter plot and line which depicts a PC being fitted on a 2-D data matrix.

.. figure:: ../_static/pca/pca.gif
    :align: center

    Making sense of PCA by fitting on a 2-D dataset (`source <https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues>`_)

Calculation of PCA
------------------

Here, we begin by explaining the usual PCA used in Quadratic Discriminant Function (QDF). This follows the introduction of that in :cite:`maeda2010subspace`.

Let, :math:`x_i` be the :math:`i^{th}` vector in our target dataset with :math:`N` samples, and :math:`\mu` be the mean :math:`\frac{1}{N}\sum_{i}x_i`. :math:`\phi` is the unit vector that represents the direction that maximizes the variance of the distribution.

Our goal is to capture the direction that maximizes the variance of the distribution, e.g. solve for :math:`\phi` with the condition that :math:`|| \phi ||=1`. This can be formulated with the introduction of an unknown multiplier :math:`\lambda` and follows:

.. math::
    J = \sum{i} (x_a - \mu, \phi)^2 - \lambda (||\phi||^2-1) \\
    = \sum_{i}\{ \sum_{j}(x_{ij} - m_j), \phi \}^2 - \lambda (\sum_{j}||\phi||^2-1)

We can solve for :math:`|| \phi ||` by setting the partial derivatives of J to zero. This leaves us with:

.. math::
    \sum_{i}(x_i - \mu, \phi)(x_i - \mu) - \lambda (2\phi) = 0

This can be rewritten further as shown below,

.. math::
    \sum_{i}(x_i - \mu)(x_i - \mu)T\phi = \lambda (2\phi)
    
If we focus on :math:`\sum_{i}(x_i - \mu)(x_i - \mu)`, we can see this is a matrix. Replacing this matrix with :math:`K` such that :math:`K=\sum_{i}(x_i - \mu)(x_i - \mu)`, we find a formula that many will find familiar.

.. math::
    K \phi = \lambda\phi

In conclusion, \phi becomes the eigen vector of covariance matrix K. As for :math:`\phi(i>0)`, the story is the same and :math:`\phi` are obtain as the eigenvectors.



PCA in the Subspace Method
--------------------------

In many cases, the reference of the subspace Method is made by PCA.

Usually, the principal components are determined as vectors which start from the center point of a classes distribution. In other words, we center the distribution according to the mean, such that the mean of the distribution will be zero. :cite:`maeda2010subspace`.

However, such mean centering is not used in the Subspace Method. The Subspace Method's PCA is different from usual PCA at this point.



References
..........
.. bibliography:: ../ref.bib