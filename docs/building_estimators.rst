Building Estimators
===================

Given a collection of fibers, we provide several utility functions to construct the integral estimators for a problem of interest. 
We think the most useful view of the world is to think about the geometric primitives of the application as arising from a sublevel set of 
an appropriately defined function. 
At first blush this may seem overpowered and unnecessarily technical as a method to describe simple shapes (and below we do discuss functionality for 
special cases), but in many applications we want to parameterize quite complicated and flexible geometries, and this unified description generalizes the 
basic shapes.

With that in mind, the most general way to build an estimator involves formulating a function whose zero-sublevel set corresponds with the shape/geometry of interest. 
Concretely, let's complete the example from Basics. 
We need to parameterize the disc as a scalar-valued function on R2 which takes on negative values within the disc, and non-negative values outside of it. 

.. code-block:: python 

   def disc_fn(x: Float[Array, "2"], disc_center: Float[Array, "2"]): 
      return np.linalg.norm(x - disc_center) - 1. 

We call this scalar-valued function which implicitly defines our shapes a 'field', and we provide estimators for the integral associated with the field (using the convention that the zero sublevel-set of the field corresponds with the interior of the shape). 
For example, to estimate pi, we could take the quotient of the estimated area of the disc and the estimated area of the sampling domain. 
As is the typical case, the way we sampled fibers means that we use the cumulative length of all the fibers we sampled as our estimated area of the sampling domain, 
this is handled within estimate_field_area. 

.. code-block:: python 

   from fibermc.estimators import estimate_field_area

   disc_area_estimate = estimate_field_area(lambda x: disc_fn(x, np.array([0.5]*2)), fibers, args) 

Again, we want to flag that this approach seems (and is) overpowered for many simple cases. 
We provide a simpler way to handle an extremely common class of shapes: polyhedra. 
To estimate the area within a polyhedron, one needs to provide the vertices of that polyhedron (i.e., the convex hull) in counter-clockwise order, after which 
it is simple to estimate the area of the polyhedron. 

.. code-block:: python 

   from fibermc.estimators import estimate_hull_area

   triangle = np.array([
      [0., 0.], 
      [1., 0.], 
      [0., 1.]
      ])
   hull_area_estimate = estimate_hull_area(fibers, triangle)

For more exploratory applications, we provide an extra utility to interoperate with Shapely. 
Note that these methods cannot be JIT compiled so performance-critical applications should call the lower-level utilies. 
Checkout examples for use of the high level Shapely interoperable functions. 

