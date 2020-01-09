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


When projecting data onto the generalized difference subspace, if the number of dimensions g is (the number of classes -1), the Fisher discrimination ratio of the projected data is maximized.

The number of dimensions g of the generalized difference subspace is determined by the number of eigenvectors extracted from W.


Constrained MSM (CMSM)
----------------------

In Constrained MSM (CMSM), a constrained subspace :math:`\mathcal{C}` is introduced. :math:`\mathcal{C}` is a ideally a subspace which includes the effective components for recognition and does not include any unnecessary components such as undersirable variation. Usually, the GDS is used as :math:`\mathcal{C}`.

Learning Phase
~~~~~~~~~~~~~~

1. Generate :math:`k` class subspaces from each class by using PCA.

Recognition Phase
~~~~~~~~~~~~~~~~~
