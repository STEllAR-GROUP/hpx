**************************
 HPX Breadth First Search
**************************

This application implements graph traversal via the breadth-first search (BFS)
algorithm. BFS starts at a root node, and examines all the neighbors of the
root node (e.g. the nodes nearest to the root). Then, the unexplored neighbors
of the nearest nodes are searched. This process is expanded recursively until
the termination condition for the search is fulfilled.

Options
-------

--n : std::size_t : 100 
    The number of nodes in the graph.

--max-num-neighbors : std::size_t : 20 
    The maximum number of neighbors.

--max-levels : std::size_t : 20 
    The maximum number of levels to traverse.

--root : std::size_t : 0 
    The root node in the graph.

--graph : std::string : g1.txt 
    The file containing the graph.

Input File Format
-----------------

printf-style format:::

    %u %u 

Description::
    
    node-index neighbor-node-index

Example Run
-----------

::

   bfs_client --threads 4 --root 4 --n 100 --graph g1.txt


