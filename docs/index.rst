.. Fibermc documentation master file, created by
   sphinx-quickstart on Tue Dec 10 14:51:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Fibermc
=======

Differentiable Monte Carlo in JAX with applications to computational geometry, differentiable simulation, and topology optimization.

- **Hardware accelerated**: our estimators and geometric kernels can target CPUs as well as accelerators (GPU, TPU) inheriting from Jax. 
- **Compatible with Jax transformations**: fully compatible with Jax transforms like ``vmap`` for vectorizing/batching and ``jit`` for just-in-time compilation. 
- **Differentiable**: estimators can be used directly with ``jax.grad``; we provide implementations for implicitly differentiating flexible geometries. 

Installation 
------------

To install the latest release of fibermc, use the following command: 

.. code-block:: console

   $ pip install fibermc

Alternatively, it can be installed from source with the following command: 

.. code-block:: console

   $ python3 -m build
   $ pip install dist/*.whl 

.. toctree::
   :maxdepth: 1
   :caption: Documentation:

   basics 
   sampling_fibers 
   building_estimators 
   differentiating_estimators

.. toctree::
   :maxdepth: 1 
   :caption: API 

   api 

.. toctree:: 
   :maxdepth: 2
   :caption: Examples 

   examples/index

Support 
-------

.. note:: 
   
   This is a research project with minimal maintenance. 

If you are having issues, file an issue on the `issue tracker <https://github.com/PrincetonLIPS/fibermc/issues>`_.

License 
-------

Fibermc is licensed under the MIT license. 

Citing 
------

If this software proves useful for you, please consider citing the associated paper that describes the underlying method in greater detail: 

.. code-block:: bibtex

   @article{fibermc,
     title={Fiber Monte Carlo},
     author={Richardson, Nick and Oktay, Deniz and Ovadia, Yaniv and Bowden, James C and Adams, Ryan P},
     journal={ICLR 2024},
     year={2024}
   }
