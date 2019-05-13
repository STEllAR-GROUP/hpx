..
    Copyright (C) 2019 Tapasweni

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _examples_parallelism:

==========================================
Parallelism: Parallelism
==========================================

Parallel Execution Properties
==============================

- The execution restrictions applicable for work items ('safe to be executed concurrently')
- In what sequence work items have to be executed
- Where work items should be executed (CPU, NUMA domain, Core, GPU)
- The parameters of the execution environment (chunk sizes, etc.)


.. figure:: /_static/images/parallelism-0.png

.. figure:: /_static/images/parallelism-1.png

.. figure:: /_static/images/parallelism-2.png

.. figure:: /_static/images/parallelism-3.png

.. figure:: /_static/images/parallelism-4.png

.. figure:: /_static/images/parallelism-5.png


Parallel Algorithms
====================


.. figure:: /_static/images/parallel_stl.png


Serial VS Parallel
===================

`Serial Version`

.. code-block:: cpp

    std::vector<double> a = ...;
		std::vector<double> b = ...;
	  std::vector<double> c = ...;
    double x = ...;

		std::transform(
    		b.begin(), b.end(), c.begin(),
    		a.begin(),
    		[x](double bb, double cc)
   		  {
        		return bb * x + cc;
   		 	}
		);


`Parallel Version`

.. code-block:: cpp

   std::vector<double> a = ...;
	 std::vector<double> b = ...;
   std::vector<double> c = ...;
   double x = ...;

   hpx::parallel::transform(
      hpx::parallel::execution::par,
      b.begin(), b.end(), c.begin(),
      a.begin(),
      [x](double bb, double cc)
      {
         return bb * x + cc;
      }
   );
