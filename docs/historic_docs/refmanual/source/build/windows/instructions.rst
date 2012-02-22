.. _windows_build_instructions:

**************
 Instructions 
**************

.. sectionauthor:: Bryce Lelbach 

0) Download the latest Subversion revision of HPX (or a stable tag). You can
   use TortoiseSVN, or the subversion client that Cygwin provides. The latest
   subversion revision can be found at:::

    https://svn.cct.lsu.edu/repos/projects/parallex/trunk/hpx

1) Create a build folder. HPX requires an out-of-tree-build. This means that you
   will be unable to run CMake in the HPX source folder.

2) Open up the CMake GUI. In the input box labelled "Where is the source code:",
   enter the full path to the source folder. In the input box labelled
   "Where to build the binaries:", enter the full path to the build folder you
   created in step 1.

3) Add CMake variable definitions (if any) by clicking the "Add Entry" button.

4) Press the "Configure" button. A window will pop up asking you which compilers
   to use. Select the x64 Visual Studio 10 compilers (they are usually the
   default if available).

5) Press "Configure" again. Repeat this step until the "Generate" button becomes
   clickable.

6) Press "Generate".

7) Open up the build folder, and double-click hpx.sln.

8) Build the INSTALL target.

