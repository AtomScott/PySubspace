Kernel Principle Component Analysis
============================

Summary
-------

Kernel Principal Component Analysis (kernel PCA) extends `principal component analysis (PCA) <PCA.html>`_ using kernel methods. KPCA allows originally linear operations of PCA to performed in a generalized euclidean space (`Reproducing kernel Hilbert space <https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space>`_).

Given an appropriate kernel, it is possible to construct a hyper plane that linearly separates the classes the in feature space. 

.. figure:: ../_static/tutorials/kpca1.png
    :align: center

    Left: Input points before KPCA. Center: Output after kernel PCA with :math:`k(\mathbf{x}^{\top}\mathbf{y}+1)^2`. Right : Output after KPCA with a Gaussian kernel. (`source <https://en.wikipedia.org/wiki/Kernel_principal_component_analysis>`_)

Theory
------

In general, kernel methods use a non-linear transformation :math:`\phi: \mathbb{R}^d \to \mathbb{R}^{'}` to handle data :math:`\mathbb{X} = [\mathbf{x}^{(1)}, .. \mathbf{x}^{(N)}]` in feature space. 

Kernel Trick
~~~~~~~~~~~~

Calculating directly in feature space is unrealistic (*exaplanation*). However, by defining a kernel function :math:`k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})=\phi(\mathbf{x}^{(i)})^{\top}, \phi(\mathbf{x}^{(j)})` to calcualte the inner product we can perform inner product calculation in the original space. Therefore greatly reducing calculation time. This approach is named the kernel trick.

Radial Basis Function Kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In many kernel based classification methods, the Radial Basis Function Kernel is used. It is defined as the following:

.. math::
    k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = exp(-\frac{||\mathbf{x}^{(i)}-\mathbf{x}^{(j)}||^2}{2\sigma ^2})

.. note::
    Subspaces methods such as KMSM and KCMSM universally use RBF as the kernel. RBF is also the default parameter for in support vector classifier in the `scikit-learn implementation <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_ .


Calculation of KPCA
-------------------

