..
    Copyright (C) 2007-2018 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_0_9_7:

===========================
|hpx| V0.9.7 (Nov 13, 2013)
===========================

We have had over 1000 commits since the last release and we have closed over 180
tickets (bugs, feature requests, etc.).

General changes
===============

* Ported HPX to BlueGene/Q
* Improved HPX support for Xeon/Phi accelerators
* Reimplemented ``hpx::bind``, ``hpx::tuple``, and ``hpx::function`` for better
  performance and better compliance with the |cpp11|. Added ``hpx::mem_fn``.
* Reworked ``hpx::when_all`` and ``hpx::when_any`` for better compliance with
  the ongoing C++ standardization effort, added heterogeneous version for those
  functions. Added ``hpx::when_any_swapped``.
* Added ``hpx::copy`` as a precursor for a migrate functionality
* Added ``hpx::get_ptr`` allowing to directly access the memory underlying a
  given component
* Added the ``hpx::lcos::broadcast``, ``hpx::lcos::reduce``, and
  ``hpx::lcos::fold`` collective operations
* Added ``hpx::get_locality_name`` allowing to retrieve the name of any of the
  localities for the application.
* Added support for more flexible thread affinity control from the HPX command
  line, such as new modes for ``--hpx:bind`` (``balanced``, ``scattered``,
  ``compact``), improved default settings when running multiple localities on
  the same node.
* Added experimental executors for simpler thread pooling and scheduling. This
  API may change in the future as it will stay aligned with the ongoing C++
  standardization efforts.
* Massively improved the performance of the HPX serialization code. Added
  partial support for zero copy serialization of array and bitwise-copyable
  types.
* General performance improvements of the code related to threads and futures.

Bug fixes (closed tickets)
==========================

Here is a list of the important tickets we closed for this release.

* :hpx-issue:`1005` - Allow one to disable array optimizations and zero copy
  optimizations for each parcelport
* :hpx-issue:`1004` - Generate new HPX logo image for the docs
* :hpx-issue:`1002` - If MPI parcelport is not available, running HPX under
  mpirun should fail
* :hpx-issue:`1001` - Zero copy serialization raises assert
* :hpx-issue:`1000` - Can't connect to a HPX application running with the MPI
  parcelport from a non MPI parcelport locality
* :hpx-issue:`999` - Optimize ``hpx::when_n``
* :hpx-issue:`998` - Fixed const-correctness
* :hpx-issue:`997` - Making serialize_buffer::data() type save
* :hpx-issue:`996` - Memory leak in hpx::lcos::promise
* :hpx-issue:`995` - Race while registering pre-shutdown functions
* :hpx-issue:`994` - thread_rescheduling regression test does not compile
* :hpx-issue:`992` - Correct comments and messages
* :hpx-issue:`991` - setcap cap_sys_rawio=ep for power profiling causes an HPX
  application to abort
* :hpx-issue:`989` - Jacobi hangs during execution
* :hpx-issue:`988` - multiple_init test is failing
* :hpx-issue:`986` - Can't call a function called "init" from "main" when using
  ``<hpx/hpx_main.hpp>``
* :hpx-issue:`984` - Reference counting tests are failing
* :hpx-issue:`983` - thread_suspension_executor test fails
* :hpx-issue:`980` - Terminating HPX threads don't leave stack in virgin state
* :hpx-issue:`979` - Static scheduler not in documents
* :hpx-issue:`978` - Preprocessing limits are broken
* :hpx-issue:`977` - Make tests.regressions.lcos.future_hang_on_get shorter
* :hpx-issue:`976` - Wrong library order in pkgconfig
* :hpx-issue:`975` - Please reopen #963
* :hpx-issue:`974` - Option pu-offset ignored in fixing_588 branch
* :hpx-issue:`972` - Cannot use MKL with HPX
* :hpx-issue:`969` - Non-existent INI files requested on the command line via
  ``--hpx:config`` do not cause warnings or errors.
* :hpx-issue:`968` - Cannot build examples in fixing_588 branch
* :hpx-issue:`967` - Command line description of ``--hpx:queuing`` seems wrong
* :hpx-issue:`966` - ``--hpx:print-bind`` physical core numbers are wrong
* :hpx-issue:`965` - Deadlock when building in Release mode
* :hpx-issue:`963` - Not all worker threads are working
* :hpx-issue:`962` - Problem with SLURM integration
* :hpx-issue:`961` - ``--hpx:print-bind`` outputs incorrect information
* :hpx-issue:`960` - Fix cut and paste error in documentation of
  get_thread_priority
* :hpx-issue:`959` - Change link to boost.atomic in documentation to point to
  boost.org
* :hpx-issue:`958` - Undefined reference to intrusive_ptr_release
* :hpx-issue:`957` - Make tuple standard compliant
* :hpx-issue:`956` - Segfault with a3382fb
* :hpx-issue:`955` - ``--hpx:nodes`` and ``--hpx:nodefiles`` do not work with
  foreign nodes
* :hpx-issue:`954` - Make order of arguments for hpx::async and hpx::broadcast
  consistent
* :hpx-issue:`953` - Cannot use MKL with HPX
* :hpx-issue:`952` - ``register_[pre_]shutdown_function`` never throw
* :hpx-issue:`951` - Assert when number of threads is greater than hardware
  concurrency
* :hpx-issue:`948` - ``HPX_HAVE_GENERIC_CONTEXT_COROUTINES`` conflicts with
  ``HPX_HAVE_FIBER_BASED_COROUTINES``
* :hpx-issue:`947` - Need MPI_THREAD_MULTIPLE for backward compatibility
* :hpx-issue:`946` - HPX does not call ``MPI_Finalize``
* :hpx-issue:`945` - Segfault with ``hpx::lcos::broadcast``
* :hpx-issue:`944` - OS X: assertion ``pu_offset_ < hardware_concurrency``
  failed
* :hpx-issue:`943` - #include <hpx/hpx_main.hpp> does not work
* :hpx-issue:`942` - Make the BG/Q work with -O3
* :hpx-issue:`940` - Use separator when concatenating locality name
* :hpx-issue:`939` - Refactor MPI parcelport to use ``MPI_Wait`` instead of
  multiple ``MPI_Test`` calls
* :hpx-issue:`938` - Want to officially access ``client_base::gid_``
* :hpx-issue:`937` - ``client_base::gid_`` should be private``
* :hpx-issue:`936` - Want doxygen-like source code index
* :hpx-issue:`935` - Build error with gcc 4.6 and Boost 1.54.0 on hpx trunk and
  0.9.6
* :hpx-issue:`933` - Cannot build HPX with Boost 1.54.0
* :hpx-issue:`932` - Components are destructed too early
* :hpx-issue:`931` - Make HPX work on BG/Q
* :hpx-issue:`930` - make git-docs is broken
* :hpx-issue:`929` - Generating index in docs broken
* :hpx-issue:`928` - Optimize ``hpx::util::static_`` for C++11 compilers
  supporting magic statics
* :hpx-issue:`924` - Make kill_process_tree (in process.py) more robust on Mac
  OSX
* :hpx-issue:`923` - Correct BLAS and RNPL cmake tests
* :hpx-issue:`922` - Cannot link against BLAS
* :hpx-issue:`921` - Implement ``hpx::mem_fn``
* :hpx-issue:`920` - Output locality with ``--hpx:print-bind``
* :hpx-issue:`919` - Correct grammar; simplify boolean expressions
* :hpx-issue:`918` - Link to hello_world.cpp is broken
* :hpx-issue:`917` - adapt cmake file to new boostbook version
* :hpx-issue:`916` - fix problem building documentation with xsltproc >= 1.1.27
* :hpx-issue:`915` - Add another TBBMalloc library search path
* :hpx-issue:`914` - Build problem with Intel compiler on Stampede (TACC)
* :hpx-issue:`913` - fix error messages in fibonacci examples
* :hpx-issue:`911` - Update OS X build instructions
* :hpx-issue:`910` - Want like to specify MPI_ROOT instead of compiler wrapper
  script
* :hpx-issue:`909` - Warning about void* arithmetic
* :hpx-issue:`908` - Buildbot for MIC is broken
* :hpx-issue:`906` - Can't use ``--hpx:bind=balanced`` with multiple MPI
  processes
* :hpx-issue:`905` - ``--hpx:bind`` documentation should describe full grammar
* :hpx-issue:`904` - Add hpx::lcos::fold and hpx::lcos::inverse_fold collective
  operation
* :hpx-issue:`903` - Add ``hpx::when_any_swapped()``
* :hpx-issue:`902` - Add ``hpx::lcos::reduce`` collective operation
* :hpx-issue:`901` - Web documentation is not searchable
* :hpx-issue:`900` - Web documentation for trunk has no index
* :hpx-issue:`898` - Some tests fail with GCC 4.8.1 and MPI parcel port
* :hpx-issue:`897` - HWLOC causes failures on Mac
* :hpx-issue:`896` - pu-offset leads to startup error
* :hpx-issue:`895` - ``hpx::get_locality_name`` not defined
* :hpx-issue:`894` - Race condition at shutdown
* :hpx-issue:`893` - ``--hpx:print-bind`` switches std::cout to hexadecimal mode
* :hpx-issue:`892` - ``hwloc_topology_load`` can be expensive -- don't call
  multiple times
* :hpx-issue:`891` - The documentation for ``get_locality_name`` is wrong
* :hpx-issue:`890` - ``--hpx:print-bind`` should not exit
* :hpx-issue:`889` - ``--hpx:debug-hpx-log=FILE`` does not work
* :hpx-issue:`888` - MPI parcelport does not exit cleanly for --hpx:print-bind
* :hpx-issue:`887` - Choose thread affinities more cleverly
* :hpx-issue:`886` - Logging documentation is confusing
* :hpx-issue:`885` - Two threads are slower than one
* :hpx-issue:`884` - is_callable failing with member pointers in C++11
* :hpx-issue:`883` - Need help with is_callable_test
* :hpx-issue:`882` - tests.regressions.lcos.future_hang_on_get does not
  terminate
* :hpx-issue:`881` - tests/regressions/block_matrix/matrix.hh won't compile with
  GCC 4.8.1
* :hpx-issue:`880` - HPX does not work on OS X
* :hpx-issue:`878` - ``future::unwrap`` triggers assertion
* :hpx-issue:`877` - "make tests" has build errors on Ubuntu 12.10
* :hpx-issue:`876` - tcmalloc is used by default, even if it is not present
* :hpx-issue:`875` - global_fixture is defined in a header file
* :hpx-issue:`874` - Some tests take very long
* :hpx-issue:`873` - Add block-matrix code as regression test
* :hpx-issue:`872` - HPX documentation does not say how to run tests with
  detailed output
* :hpx-issue:`871` - All tests fail with "make test"
* :hpx-issue:`870` - Please explicitly disable serialization in classes that
  don't support it
* :hpx-issue:`868` - boost_any test failing
* :hpx-issue:`867` - Reduce the number of copies of ``hpx::function`` arguments
* :hpx-issue:`863` - Futures should not require a default constructor
* :hpx-issue:`862` - value_or_error shall not default construct its result
* :hpx-issue:`861` - ``HPX_UNUSED`` macro
* :hpx-issue:`860` - Add functionality to copy construct a component
* :hpx-issue:`859` - ``hpx::endl`` should flush
* :hpx-issue:`858` - Create ``hpx::get_ptr<>`` allowing to access component
  implementation
* :hpx-issue:`855` - Implement ``hpx::INVOKE``
* :hpx-issue:`854` - ``hpx/hpx.hpp`` does not include
  ``hpx/include/iostreams.hpp``
* :hpx-issue:`853` - Feature request: null future
* :hpx-issue:`852` - Feature request: Locality names
* :hpx-issue:`851` - ``hpx::cout`` output does not appear on screen
* :hpx-issue:`849` - All tests fail on OS X after installing
* :hpx-issue:`848` - Update OS X build instructions
* :hpx-issue:`846` - Update hpx_external_example
* :hpx-issue:`845` - Issues with having both debug and release modules in the
  same directory
* :hpx-issue:`844` - Create configuration header
* :hpx-issue:`843` - Tests should use CTest
* :hpx-issue:`842` - Remove buffer_pool from MPI parcelport
* :hpx-issue:`841` - Add possibility to broadcast an index with
  hpx::lcos::broadcast
* :hpx-issue:`838` - Simplify ``util::tuple``
* :hpx-issue:`837` - Adopt boost::tuple tests for ``util::tuple``
* :hpx-issue:`836` - Adopt boost::function tests for ``util::function``
* :hpx-issue:`835` - Tuple interface missing pieces
* :hpx-issue:`833` - Partially preprocessing files not working
* :hpx-issue:`832` - Native papi counters do not work with wild cards
* :hpx-issue:`831` - Arithmetics counter fails if only one parameter is given
* :hpx-issue:`830` - Convert hpx::util::function to use new scheme for
  serializing its base pointer
* :hpx-issue:`829` - Consistently use ``decay<T>`` instead of ``remove_const<
  remove_reference<T>>``
* :hpx-issue:`828` - Update future implementation to N3721 and N3722
* :hpx-issue:`827` - Enable MPI parcelport for bootstrapping whenever
  application was started using mpirun
* :hpx-issue:`826` - Support command line option ``--hpx:print-bind`` even if
  ``--hpx::bind`` was not used
* :hpx-issue:`825` - Memory counters give segfault when attempting to use thread
  wild cards or numbers only total works
* :hpx-issue:`824` - Enable lambda functions to be used with
  hpx::async/hpx::apply
* :hpx-issue:`823` - Using a hashing filter
* :hpx-issue:`822` - Silence unused variable warning
* :hpx-issue:`821` - Detect if a function object is callable with given
  arguments
* :hpx-issue:`820` - Allow wildcards to be used for performance counter names
* :hpx-issue:`819` - Make the AGAS symbolic name registry distributed
* :hpx-issue:`818` - Add future::then() overload taking an executor
* :hpx-issue:`817` - Fixed typo
* :hpx-issue:`815` - Create an lco that is performing an efficient broadcast of
  actions
* :hpx-issue:`814` - Papi counters cannot specify thread#* to get the counts for
  all threads
* :hpx-issue:`813` - Scoped unlock
* :hpx-issue:`811` - simple_central_tuplespace_client run error
* :hpx-issue:`810` - ostream error when << any objects
* :hpx-issue:`809` - Optimize parcel serialization
* :hpx-issue:`808` - HPX applications throw exception when executed from the
  build directory
* :hpx-issue:`807` - Create performance counters exposing overall AGAS
  statistics
* :hpx-issue:`795` - Create timed make_ready_future
* :hpx-issue:`794` - Create heterogeneous ``when_all``/``when_any``/etc.
* :hpx-issue:`721` - Make HPX usable for Xeon Phi
* :hpx-issue:`694` - CMake should complain if you attempt to build an example
  without its dependencies
* :hpx-issue:`692` - SLURM support broken
* :hpx-issue:`683` - python/hpx/process.py imports epoll on all platforms
* :hpx-issue:`619` - Automate the doc building process
* :hpx-issue:`600` - GTC performance broken
* :hpx-issue:`577` - Allow for zero copy serialization/networking
* :hpx-issue:`551` - Change executable names to have debug postfix in Debug
  builds
* :hpx-issue:`544` - Write a custom .lib file on Windows pulling in hpx_init and
  hpx.dll, phase out hpx_init
* :hpx-issue:`534` - ``hpx::init`` should take functions by ``std::function``
  and should accept all forms of hpx_main
* :hpx-issue:`508` - FindPackage fails to set FOO_LIBRARY_DIR
* :hpx-issue:`506` - Add cmake support to generate ini files for external
  applications
* :hpx-issue:`470` - Changing build-type after configure does not update boost
  library names
* :hpx-issue:`453` - Document ``hpx_run_tests.py``
* :hpx-issue:`445` - Significant performance mismatch between MPI and HPX in SMP
  for allgather example
* :hpx-issue:`443` - Make docs viewable from build directory
* :hpx-issue:`421` - Support multiple HPX instances per node in a batch
  environment like PBS or SLURM
* :hpx-issue:`316` - Add message size limitation
* :hpx-issue:`249` - Clean up locking code in big boot barrier
* :hpx-issue:`136` - Persistent CMake variables need to be marked as cache
  variables

