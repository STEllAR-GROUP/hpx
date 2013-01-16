.. Copyright (c) 2007-2013 Louisiana State University

   Distributed under the Boost Software License, Version 1.0. (See accompanying
   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

*****
 HPX
*****

HPX is a general purpose C++ runtime system for parallel and distributed
applications of any scale. Even if that's quite a mouthful, we mean every
word of it!

The goal of HPX is to create a high quality, freely available, open source
implementation of the ParalleX model for conventional systems, such as
classic Linux based Beowulf clusters or multi-socket highly parallel SMP
nodes. At the same time, we want to have a very modular and well designed
runtime system architecture which would allow us to port our implementation
onto new computer system architectures. We want to use real world applications
to drive the development of the runtime system, coining out required
functionalities and converging onto a stable API which will provide a
smooth migration path for developers. The API exposed by HPX is modelled
after the interfaces defined by the C++11 ISO standard and adheres to the
programming guidelines used by the Boost collection of C++ libraries.

****************************
What's so special about HPX?
****************************

* HPX exposes an uniform, standards-oriented API for ease of programming
  parallel and distributed applications.
* It enables programmers to write fully asynchronous  code using hundreds
  of millions of threads.
* HPX provides unified syntax and semantics for local and remote operations.
* HPX makes concurrency manageable with dataflow and future based
  synchronization.
* It implements a rich set of runtime services supporting a broad range of
  use cases.
* It is designed to solve problems conventionally considered to be
  scaling-impaired.
* HPX has been designed and developed for systems of any scale, from
  hand-held devices to very large scale systems.
* It is the first fully functional implementation of the ParalleX execution
  model.
* HPX is published under a liberal open-source license and has a open,
  active, and thriving developer community.


The documentation for the latest release of HPX (currently V0.9.5) can be
`found here <http://stellar.cct.lsu.edu/files/hpx_0.9.5/html/index.html>`_.

.. Additionally, we regularily upload the current status of the documentation
   (which is being worked on as we speak)
   `here <http://stellar.cct.lsu.edu/files/hpx_master/docs/index.html>`_.

If you plan to use HPX we suggest to start with the latest released version
(currently HPX V0.9.5) which can be `downloaded here <http://stellar.cct.lsu.edu/downloads/>`_.

If you would like to work with the cutting edge version from this repository
we suggest following the current health status of the master branch by looking at
our `contiguous integration results website <http://ithaca.cct.lsu.edu/waterfall>`_.
While we try to keep the master branch stable and usable, sometimes new bugs
trick their way into the code base - you have been warned!

In any case, if you happen to run into problems we very much encourage and appreciate
any issue reports through the `issue tracker for this Github project
<http://github.com/STEllAR-GROUP/hpx/issues>`_.

********************
 Build Instructions
********************

All of HPX is distributed under the Boost Software License,
Version 1.0 (See accompanying file LICENSE_1_0.txt or an online copy available
`here <http://www.boost.org/LICENSE_1_0.txt>`_).

Before starting to build HPX, please read about the
`prerequisites <http://stellar.cct.lsu.edu/files/hpx_0.9.0/docs/hpx/tutorial/getting_started.html>`_.

Linux
-----

1) Clone the master HPX git repository (or a stable tag)::

    $ git clone git://github.com/STEllAR-GROUP/hpx.git

2) Create a build directory. HPX requires an out-of-tree build. This means you
   will be unable to run CMake in the HPX source directory::

    $ cd hpx
    $ mkdir my_hpx_build
    $ cd my_hpx_build

3) Invoke CMake from your build directory, pointing the CMake driver to the root
   of your HPX source tree::

    $ cmake -DBOOST_ROOT=/your_boost_directory [other CMake variable definitions] /path/to/hpx/source/tree

4) Invoke GNU make. If you are on a machine with multiple cores (very likely),
   add the -jN flag to your make invocation, where N is the number of nodes
   on your machine plus one::

    $ gmake -j5

5) To complete the build and install HPX::

    $ gmake install

Mac OSX
-------

1) Install a recent version of LLVM and Clang.
   In order to build hpx you will need a fairly recent version of Clang
   (at least version 3.2 of Clang and LLVM). For more instructions please 
   see http://clang.llvm.org/get_started.html.

   If you're using Homebrew, ``brew install llvm --with-clang`` will do the trick.
   This will install Clang V3.2 into ``/usr/local/bin``.

2) Visit http://libcxx.llvm.org/ to get the latest version of the "libc++" C++ 
   standard library. You need to use the trunk version; what's currently bundled
   with XCode or OS X aren't quite there yet. You can follow the steps in
   http://libcxx.llvm.org/ if you choose, but here's briefly how it could be built::

      cd /path/to
      git clone http://llvm.org/git/libcxx.git
      cd libcxx/lib
      CXX=clang++-3.2 CC=clang-3.2 TRIPLE=-apple- ./buildit

   The library is then found in ``/path/to/libcxx/include`` and
   ``/path/to/libcxx/lib``, respectively.

3) Build (and install) a recent version of Boost, using Clang and libc++::
   To build Boost with Clang and make it link to libc++ as standard library,
   you'll need to set up the following in your Boost ``~/user-config.jam``
   file::

      # user-config.jam (put this file into your home directory)
      # ...
      # Clang 3.2
      using clang
        : 3.2
        : "/usr/local/bin/clang++"
        : <cxxflags>"-std=c++11 -stdlib=libc++ -isystem /path/to/libcxx/include"
          <linkflags>"-stdlib=libc++ -L/path/to/libcxx/lib"
        ;

   You can then use this as your build command::

      b2 --build-dir=/tmp/build-boost --layout=versioned toolset=clang-3.2 install -j5

4) Clone the master HPX git repository (or a stable tag)::

    $ git clone git://github.com/STEllAR-GROUP/hpx.git

5) Build HPX, finally::

      $ cd hpx
      $ mkdir my_hpx_build
      $ cd my_hpx_build

   To build with Clang 3.2, execute::

      $ cmake /path/to/hpx/source/tree \
           -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++ \
           -DCMAKE_C_COMPILER=/usr/local/bin/clang-3.2 \
           -DBOOST_ROOT=/your_boost_directory \
           -DCMAKE_CXX_FLAGS="-isystem /path/to/libcxx/include" \
           -DLINK_FLAGS="-L /path/to/libcxx/lib"
      $ make -j5

6) To complete the build and install HPX::

    $ make install

Windows
-------

1) Clone the master HPX git repository (or a stable tag). You can use
   TortoiseGIT, or the git client that Cygwin provides. The git repository can
   be found at::

    git://github.com/STEllAR-GROUP/hpx.git

2) Create a build folder. HPX requires an out-of-tree-build. This means that you
   will be unable to run CMake in the HPX source folder.

3) Open up the CMake GUI. In the input box labelled "Where is the source code:",
   enter the full path to the source folder. In the input box labelled
   "Where to build the binaries:", enter the full path to the build folder you
   created in step 1.

4) Add CMake variable definitions (if any) by clicking the "Add Entry" button.
   Most probably you will need to at least add the directory where Boost is
   located as BOOST_ROOT.

5) Press the "Configure" button. A window will pop up asking you which compilers
   to use. Select the x64 Visual Studio 10 compiler (Visual Studio 2012 is
   supported as well).

6) If the "Generate" button is not clickable, press "Configure" again. Repeat
   this step until the "Generate" button becomes clickable.

7) Press "Generate".

8) Open up the build folder, and double-click hpx.sln.

9) Build the INSTALL target.

