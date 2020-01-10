=============
API Reference
=============

This is the class and function reference of scikit-learn. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.
For reference on concepts repeated across the API, see :ref:`glossary`.

Base classes
------------
.. currentmodule:: cvt

.. autosummary::
   :toctree: generated

    models.base_class.SMBase
    models.base_class.KernelSMBase
    models.base_class.ConstrainedSMBase
    models.base_class.KernelCSMBase
    models.base_class.MSMInterface

Model classes
------------

.. autosummary::
   :toctree: generated
   :template: class.rst

    models.SubspaceMethod
    models.MutualSubspaceMethod
    models.KernelMSM
    models.ConstrainedMSM
    models.KernelCMSM

Utils classes
------------
.. currentmodule:: cvt

.. autosummary::
   :toctree: generated

    utils.base
    utils.evaluation
    utils.kernel_functions
