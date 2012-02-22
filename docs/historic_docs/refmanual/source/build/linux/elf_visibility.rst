.. _linux_elf_visibility:

******************************
 ELF Visibility Optimizations 
******************************

.. sectionauthor:: Bryce Lelbach 

In |elf|_ binary objects, all symbols are exported by default. This substantially
increases the size of ELF binaries. On compilers which expose a usable interface
for controlling C++ symbol visibility, HPX will only export the symbols needed to
use HPX in release builds.

Currently, only |gcc|_ properly supports ELF symbol visibility for C++. This
support is only functional in newer versions of GCC (specifically, version 4.4.*
and up). If you are attempting to build HPX with an older version of GCC, you
will have to disable this optimization. This can be done by setting the CMake
variable HPX_ELF_HIDDEN_VISIBILITY to off.

