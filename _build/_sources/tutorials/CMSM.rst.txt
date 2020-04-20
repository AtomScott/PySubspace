Constrained Subspace Method
===========================

Summary
-------

The Constrained Mutual Subspace Method (CMSM) is an extension of the Mutual Subspace Method (MSM) :cite:`fukui2005face`. In CMSM, we project the input subspace and the reference subspace onto the General Difference Subspace (GDS). This step is useful to extract effective features for classification :cite:`Nishiyama2005`.  

.. figure:: ../_static/tutorials/CMSM.png
    :align: center

    Fig.1

General Difference Subspace (GDS)
---------------------------------

A difference space can be defined between two subspaces, just as a difference vector exists between two vectors (See fig 2.). The difference space is a subspace containing the difference vector :math:`d_i` between the :math:`i_{th}` canonical vectors of subspaces :math:`\mathcal{P}` and :math:`\mathcal{Q}`, i.e. :math:`u_i` and :math:`v_i`.

The General Difference Subspace (GDS) is not the differece subspace between two only subspaces, but instead a differece subspace of multiple subspaces :cite:`Fukui2015`.

A projection to the GDS usually results in an increase of orthoganality between subspaces. Furthermore, the projection of vector data onto the GDS can increase the Fisher discrimination ratio, and can perform feature extraction effective for discrimination.

.. figure:: ../_static/tutorials/CMSM_concept.png
    :align: center
    
    Fig 2. `source <https://www.researchgate.net/publication/220757276_Face_Recognition_Using_Multi-viewpoint_Patterns_for_Robot_Vision>`_

Calculation of GDS
~~~~~~~~~~~~~~~~~~

Let :math:`\Phi^C=[\phi_1^C,\phi_d^C]` be the :math:`d`-dimensional orthogonal bais vectors of the subspace of class :math:`C`. The GDS can be obtained by applying eigendecomposition to the following matrix :math:`G`.

.. math::
    G = \sum^C_{c=1}\sum^d_{i=1}\phi^c_i\phi_i^{c\top}
    G = W \Lambda W

When projecting data onto the generalized difference subspace, if the number of dimensions g is (the number of classes -1), the Fisher discrimination ratio of the projected data is maximized.

The number of dimensions g of the generalized difference subspace is determined by the number of eigenvectors extracted from W.


Constrained MSM (CMSM)
----------------------

In Constrained MSM (CMSM), a constrained subspace :math:`\mathcal{C}` is introduced. :math:`\mathcal{C}` is a ideally a subspace which includes the effective components for recognition and does not include any unnecessary components such as undersirable variation. Usually, the GDS is used as :math:`\mathcal{C}`.

Learning Phase
~~~~~~~~~~~~~~

1. Generate :math:`k` class subspaces from each class by using PCA.
2. Generate a GDS using the generate class subspaces.
3. Project the training data onto the GDS.
4. Generate :math:`k` class subspaces using the projected data and PCA.  

Recognition Phase
~~~~~~~~~~~~~~~~~
1. Project the input data onto the GDS abd generate the input subspace :math:`\mathcal{P}`.
2. Calculate :math:`S` (or :math:`\tilde{S}`) between :math:`\mathcal{P}` and each subspace :math:`\mathcal{Q}`. 
3. Classify :math:`\mathcal{P}` into the class where :math:`S` (or :math:`\tilde{S}`) was calculated to be the highest.

Also, you may add a rejection thershold on :math:`S` (or :math:`\tilde{S}`) to reject classifications with low similarity.
