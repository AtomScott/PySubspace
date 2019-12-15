Subspace Method
===============

Theory
------

The Subspace method Assume an input vector p and k class subspaces in f- dimensional vector space. The similarity S of the pattern vector p to the i-th class is defined based on either of the length of the projected input vector pˆ on the i-th reference subspace [3] or the minimum angle [4] between the input vector p and the i-th class subspace as shown in Fig.1(a). 

The length of an input vector p is often normalized to 1.0. In this case these two criteria coincide. In the following explanation, therefore, the angle-based similarity S defined by the following equation will be used:

S = cos2θ = ∑dq (p · ϕi)2 i=1 ||p||2, (1) 

where dq is the dimension of the class subspace, and ϕi is the i-th f-dimensional orthogonal normal basis vector of the class subspace, which are obtained from applying the principal component analysis (PCA) to a set of patterns of the class. Concretely, these orthonormal basis vectors can be obtained as the eigenvectors of the correlation matrix
i=1 xix⊤ of the class.

Process flow of the SM The whole process of the SM consists of a learning phase and a recognition phase.
In the learning phase All k class dq-dimensional subspaces are generated from a set of pattern vectors of each class by using PCA.
In the recognition phase The similarities S of an input vector p to all the k class subspaces are calculated by using Equation (1). Then, the input vector is classified into the class of the class subspace with highest similarity. If the highest similarity is lower than a threshold value fixed in advance, the input vector is classified into a reject class



