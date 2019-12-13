Subspace Methods at a Glance
============================

.. figure:: ../_static/subspace_tree.png
   :align: center
   
   Subspace methods

Subspace Analysis
----------------------

Subspace analysis in computer vision is a generic name to describe a general framework for comparison and classification of subspaces. A typical approach in subspace analysis is the subspace method (SM) that classify an input pattern vector into several classes based on the minimum distance or angle between the input pattern vector and each class subspace, where a class subspace corresponds to the distribution of pattern vectors of the class in high dimensional vector space. 



We have been extensively studied the theory and application of subspace analysis for two decades, and developed various types of subspace based method as shown in the left figure.

A typical approach in subspace analysis is the subspace method (SM) that classify an input pattern vector into several classes based on the minimum distance or angle between the input pattern vector and each class subspace, where a class subspace corresponds to the distribution of pattern vectors of the class in high dimensional vector space. 

SM was developed by two Japanese researchers, Watanabe and Iijima around 1970, independently. Watanabe and Iijima named their methods the CLAFIC and the multiple similarity method, respectively. 

Furthermore, SM has been extended to mutual subspace method (MSM) by replacing an input vector with an input subspace, where the similarity between the input and reference subspace is measured by the canonical angles between them. 

MSM has well known as one of the most natural and effective classification method for image-sets in pattern recognition and computer vision. 

We have recently been interested in the theoretical aspect of feature extraction by projecting onto a generalized difference subspace(GDS), which is called GDS projection. 

References
..........

.. [LeCun98] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.
    Gradient-based learning applied to document recognition. Proceedings of the
    IEEE, 86(11), 2278–2324, 1998.
.. [Simonyan14] Simonyan, K. and Zisserman, A., Very Deep Convolutional
    Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556,
    2014.
.. [He16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual
    Learning for Image Recognition. The IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), pp. 770-778, 2016.