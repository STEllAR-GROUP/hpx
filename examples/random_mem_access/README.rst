**********************************
 HPX Random Memory Access Example 
**********************************

This example displays one method of implementing a distributed array with
HPX. In this example, each array element is a first class object (e.g., globally
addressable). The array is mutable and elements can be randomly accessed in O(N)
amortized time complexity. In this example, we create an array of integers,
perform a number of updates to random elements of the array, and then destroy
the array.

Options
-------

--array-size : std::size_t : 8
    The size of array.

--iterations : std::size_t : 16
    The number of lookups to perform.

--seed : std::size_t : 0
    The seed for the pseudo random number generator (if 0, a seed 
    is choosen based on the current system time).

Example Run
-----------

::

   random_mem_access_client --array-size 64 --iterations 4096

