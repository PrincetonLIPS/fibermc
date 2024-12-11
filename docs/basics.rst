Basics
======

Parametric Discontinuities
--------------------------
Computing, or more often estimating, the value of an integral is a ubiquitous kernel across computational science and engineering. For instance, in graphics, the shaders underlying rasterizing rendering engines compute the value of a spatial integral over a collection of geometric primitives to produce an image. 
Modeling the generative processes associated with applications like this involves integration in the forward model (primitives to image, for instance), which contain **parametric discontinuities**. 

Mathematically, this means the integrand contains an expression which is discontinuous (with respect to a parameter in the model). 
As a simple concrete case, consider two polygons :math:`P_1, P_2 \in \Omega \subseteq \mathbb{R}^2`; squares, for instance. 
Suppose the location of :math:`P_1` is fixed, but we can choose a scalar :math:`\theta \in \mathbb{R}` by which to translate :math:`P_2` along the vector :math:`(-1, -1)^T` with the goal of maximizing the area of intersection between the two shapes. 
This corresponds with an optimization problem whose objective :math:`\mathcal{L}` is an integral with parametric discontinuity (with respect to :math:`\theta`). 

.. math::

   \mathcal{L}(\theta) = \int_{\Omega} \; \mathbb{I}[\mathbf{x} \in (P_1 \cap P_2(\theta))] \; d\mathbf{x}

Intuitively, integrals like this arise from, e.g., hard spatial boundaries and collisions between objects when simulating physics, shadowing and overlap phenomena in a rendering pipeline, or the discrete geometry that often arises in topology optimization and inverse design. 

With the increasing popularity of machine learning systems that are tightly coupled with physical simulation, rendering, and other geometric applications, these integrals are surprisingly common. 
We could use simple Monte Carlo to estimate the value of these integrals, but we typically want to reason backwards about the parameters (using the derivatives :math:`\nabla_{\theta}\mathcal{L}`), 
and simple Monte Carlo results in a non-differentiable estimate, even though :math:`\mathcal{L}` is differentiable in exact form. 

The prospect of automatically differentiating these programs with respect to the parameters is enticing, but conventional autodiff systems do not natively compute derivatives of these functions correctly. 
Fibermc provides implementations of a method called **Fiber Monte Carlo** (FMC), a differentiable variant of the simple Monte Carlo estimator. 
We provide generic, domain-independent estimators that can be used across a variety of applications, which we showcase via examples. 
