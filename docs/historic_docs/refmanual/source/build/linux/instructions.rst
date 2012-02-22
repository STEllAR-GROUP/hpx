.. _linux_build_instructions:

**************
 Instructions 
**************

.. sectionauthor:: Bryce Lelbach 

0) Download the latest Subversion revision of HPX (or a stable tag):::

    $ svn co https://svn.cct.lsu.edu/repos/projects/parallex/trunk/hpx hpx

1) Create a build directory. HPX requires an out-of-tree build. This means you
   will be unable to run CMake in the HPX source tree.::
  
    $ cd hpx
    $ mkdir my_hpx_build
    $ cd my_hpx_build

2) Invoke CMake from your build directory, pointing the CMake driver to the root
   of your HPX source tree.::

    $ cmake [CMake variable definitions] /path/to/source/tree 

3) Invoke GNU make. If you are on a box with multiple cores (very likely),
   add the -jN flag to your make invocation, where N is the number of nodes
   on your machine plus one.::

    $ gmake -j49
 
4) To complete the build and install HPX:::

    $ gmake install

