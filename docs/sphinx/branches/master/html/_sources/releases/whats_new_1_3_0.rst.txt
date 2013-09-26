..
    Copyright (C) 2007-2019 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_3_0:

===========================
|hpx| V1.3.0 (May 23, 2019)
===========================

General changes
===============

* Performance improvements: the schedulers have significantly reduced overheads
  from removing false sharing and the parallel executor has been updated to
  create fewer futures.
* HPX now defaults to not turning on networking when running on one locality.
  This means that you can run multiple instances on the same system without
  adding command line options.
* Multiple issues reported by Clang sanitizers have been fixed.
* We have added (back) single-page HTML documentation and PDF documentation.
* We have started modularizing the HPX library. This is useful both for
  developers and users. In the long term users will be able to consume only
  parts of the HPX libraries if they do not require all the functionality that
  HPX currently provides.
* We have added an implementation of ``function_ref``.
* The ``barrier`` and ``latch`` classes have gained a few additional member
  functions.

Breaking changes
================

* Executable and library targets are now created without the ``_exe`` and
  ``_lib`` suffix respectively. For example, the target ``1d_stencil_1_exe`` is
  now simply called ``1d_stencil_1``.
* We have removed the following deprecated functionality: ``queue``,
  ``scoped_unlock``, and support for input iterators in algorithms.
* We have turned off the compatibility layer for ``unwrapped`` by default. The
  functionality will be removed in the next release. The option can still be
  turned on using the |cmake|_ option ``HPX_WITH_UNWRAPPED_SUPPORT``. Likewise,
  ``inclusive_scan`` compatibility overloads have been turned off by default.
  They can still be turned on with ``HPX_WITH_INCLUSIVE_SCAN_COMPATIBILITY``.
* The minimum compiler and dependency versions have been updated. We now support
  GCC from version 5 onwards, Clang from version 4 onwards, and Boost from
  version 1.61.0 onwards.
* The headers for preprocessor macros have moved as a result of the
  functionality being moved to a separate module. The old headers are deprecated
  and will be removed in a future version of HPX. You can turn off the warnings
  by setting ``HPX_PREPROCESSOR_WITH_DEPRECATION_WARNINGS=OFF`` or turn off the
  compatibility headers completely with
  ``HPX_PREPROCESSOR_WITH_COMPATIBILITY_HEADERS=OFF``.

Closed issues
=============

* :hpx-issue:`3863` - shouldn't "-faligned-new" be a usage requirement?
* :hpx-issue:`3841` - Build error with msvc 19 caused by SFINAE and C++17
* :hpx-issue:`3836` - master branch does not build with idle rate counters
  enabled
* :hpx-issue:`3819` - Add debug suffix to modules built in debug mode
* :hpx-issue:`3817` - ``HPX_INCLUDE_DIRS`` contains non-existent directory
* :hpx-issue:`3810` - Source groups are not created for files in modules
* :hpx-issue:`3805` - HPX won't compile with ``-DHPX_WITH_APEX=TRUE``
* :hpx-issue:`3792` - Barrier Hangs When Locality Zero not included
* :hpx-issue:`3778` - Replace ``throw()`` with ``noexcept``
* :hpx-issue:`3763` - configurable sort limit per task
* :hpx-issue:`3758` - dataflow doesn't convert ``future<future<T>>`` to
  ``future<T>``
* :hpx-issue:`3757` - When compiling undefined reference to
  ``hpx::hpx_check_version_1_2`` HPX V1.2.1, Ubuntu 18.04.01 Server Edition
* :hpx-issue:`3753` - ``--hpx:list-counters=full`` crashes
* :hpx-issue:`3746` - Detection of MPI with pmix
* :hpx-issue:`3744` - Separate spinlock from same cacheline as internal data for
  all LCOs
* :hpx-issue:`3743` - hpxcxx's shebang doesn't specify the python version
* :hpx-issue:`3738` - Unable to debug parcelport on a single node
* :hpx-issue:`3735` - Latest master: Can't compile in MSVC
* :hpx-issue:`3731` - ``util::bound`` seems broken on Clang with older libstdc++
* :hpx-issue:`3724` - Allow to pre-set command line options through environment
* :hpx-issue:`3723` - examples/resource_partitioner build issue on master branch
  / ubuntu 18
* :hpx-issue:`3721` - faced a building error
* :hpx-issue:`3720` - Hello World example fails to link
* :hpx-issue:`3719` - pkg-config produces invalid output: ``-l-pthread``
* :hpx-issue:`3718` - Please make the python executable configurable through
  cmake
* :hpx-issue:`3717` - interested to contribute to the organisation
* :hpx-issue:`3699` - Remove 'HPX runtime' executable
* :hpx-issue:`3698` - Ignore all locks while handling asserts
* :hpx-issue:`3689` - Incorrect and inconsistent website structure
  `<http://stellar.cct.lsu.edu/downloads/>`_.
* :hpx-issue:`3681` - Broken links on
  `<http://stellar.cct.lsu.edu/2015/05/hpx-archives-now-on-gmane/>`_
* :hpx-issue:`3676` - HPX master built from source, cmake fails to link main.cpp
  example in docs
* :hpx-issue:`3673` - HPX build fails with ``std::atomic`` missing error
* :hpx-issue:`3670` - Generate PDF again from documentation (with Sphinx)
* :hpx-issue:`3643` - Warnings when compiling HPX 1.2.1 with gcc 9
* :hpx-issue:`3641` - Trouble with using ranges-v3 and ``hpx::parallel::reduce``
* :hpx-issue:`3639` - ``util::unwrapping`` does not work well with member
  functions
* :hpx-issue:`3634` - The build fails if ``shared_future<>::then`` is called
  with a thread executor
* :hpx-issue:`3622` - VTune Amplifier 2019 not working with ``use_itt_notify=1``
* :hpx-issue:`3616` - HPX Fails to Build with CUDA 10
* :hpx-issue:`3612` - False sharing of scheduling counters
* :hpx-issue:`3609` - executor_parameters timeout with gcc <= 7 and Debug mode
* :hpx-issue:`3601` - Misleading error message on power pc for rdtsc and rdtscp
* :hpx-issue:`3598` - Build of some examples fails when using Vc
* :hpx-issue:`3594` - Error: The number of OS threads requested (20) does not
  match the number of threads to bind (12): HPX(bad_parameter)
* :hpx-issue:`3592` - Undefined Reference Error
* :hpx-issue:`3589` - include could not find load file: HPX_Utils.cmake
* :hpx-issue:`3587` - HPX won't compile on POWER8 with Clang 7
* :hpx-issue:`3583` - Fedora and openSUSE instructions missing on "Distribution
  Packages" page
* :hpx-issue:`3578` - Build error when configuring with
  ``HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT=ON``
* :hpx-issue:`3575` - Merge openSUSE reproducible patch
* :hpx-issue:`3570` - Update HPX to work with the latest VC version
* :hpx-issue:`3567` - Build succeed and make failed for ``hpx:cout``
* :hpx-issue:`3565` - Polymorphic simple component destructor not getting called
* :hpx-issue:`3559` - 1.2.0 is missing from download page
* :hpx-issue:`3554` - Clang 6.0 warning of hiding overloaded virtual function
* :hpx-issue:`3510` - Build on ppc64 fails
* :hpx-issue:`3482` - Improve error message when ``HPX_WITH_MAX_CPU_COUNT`` is
  too low for given system
* :hpx-issue:`3453` - Two HPX applications can't run at the same time.
* :hpx-issue:`3452` - Scaling issue on the change to 2 NUMA domains
* :hpx-issue:`3442` - HPX set_difference, set_intersection failure cases
* :hpx-issue:`3437` - Ensure parent_task pointer when child task is created and
  child/parent are on same locality
* :hpx-issue:`3255` - Suspension with lock for ``--hpx:list-component-types``
* :hpx-issue:`3034` - Use C++17 structured bindings for serialization
* :hpx-issue:`2999` - Change thread scheduling use of ``size_t`` for thread
  indexing

Closed pull requests
====================

* :hpx-pr:`3865` - adds hpx_target_compile_option_if_available
* :hpx-pr:`3864` - Helper functions that are useful in numa binding and testing
  of allocator
* :hpx-pr:`3862` - Temporary fix to local_dataflow_boost_small_vector test
* :hpx-pr:`3860` - Add cache line padding to intermediate results in for loop
  reduction
* :hpx-pr:`3859` - Remove HPX_TLL_PUBLIC and HPX_TLL_PRIVATE from CMake files
* :hpx-pr:`3858` - Add compile flags and definitions to modules
* :hpx-pr:`3851` - update hpxmp release tag to v0.2.0
* :hpx-pr:`3849` - Correct BOOST_ROOT variable name in quick start guide
* :hpx-pr:`3847` - Fix attach_debugger configuration option
* :hpx-pr:`3846` - Add tests for libs header tests
* :hpx-pr:`3844` - Fixing source_groups in preprocessor module to properly
  handle compatibility headers
* :hpx-pr:`3843` - This fixes the launch_process/launched_process pair of tests
* :hpx-pr:`3842` - Fix macro call with ITTNOTIFY enabled
* :hpx-pr:`3840` - Fixing SLURM environment parsing
* :hpx-pr:`3837` - Fixing misplaced #endif
* :hpx-pr:`3835` - make all latch members protected for consistency
* :hpx-pr:`3834` - Disable transpose_block_numa example on CircleCI
* :hpx-pr:`3833` - make latch counter_ protected for deriving latch in hpxmp
* :hpx-pr:`3831` - Fix CircleCI config for modules
* :hpx-pr:`3830` - minor fix: option HPX_WITH_TEST was not working correctly
* :hpx-pr:`3828` - Avoid for binaries that depend on HPX to directly link
  against internal modules
* :hpx-pr:`3827` - Adding shortcut for ``hpx::get_ptr<>(sync, id)`` for a local,
  non-migratable objects
* :hpx-pr:`3826` - Fix and update modules documentation
* :hpx-pr:`3825` - Updating default APEX version to 2.1.3 with HPX
* :hpx-pr:`3823` - Fix pkgconfig libs handling
* :hpx-pr:`3822` - Change includes in hpx_wrap.cpp to more specific includes
* :hpx-pr:`3821` - Disable barrier_3792 test when networking is disabled
* :hpx-pr:`3820` - Assorted CMake fixes
* :hpx-pr:`3815` - Removing left-over debug output
* :hpx-pr:`3814` - Allow setting default scheduler mode via the configuration
  database
* :hpx-pr:`3813` - Make the deprecation warnings issued by the old pp headers
  optional
* :hpx-pr:`3812` - Windows requires to handle symlinks to directories
  differently from those linking files
* :hpx-pr:`3811` - Clean up PP module and library skeleton
* :hpx-pr:`3806` - Moving include path configuration to before APEX
* :hpx-pr:`3804` - Fix latch
* :hpx-pr:`3803` - Update hpxcxx to look at lib64 and use python3
* :hpx-pr:`3802` - Numa binding allocator
* :hpx-pr:`3801` - Remove duplicated includes
* :hpx-pr:`3800` - Attempt to fix Posix context switching after lazy init
  changes
* :hpx-pr:`3798` - count and count_if accepts different iterator types
* :hpx-pr:`3797` - Adding a couple of ``override`` keywords to overloaded
  virtual functions
* :hpx-pr:`3796` - Re-enable testing all schedulers in shutdown_suspended_test
* :hpx-pr:`3795` - Change ``std::terminate`` to std::abort in ``SIGSEGV``
  handler
* :hpx-pr:`3794` - Fixing #3792
* :hpx-pr:`3793` - Extending migrate_polymorphic_component unit test
* :hpx-pr:`3791` - Change ``throw()`` to ``noexcept``
* :hpx-pr:`3790` - Remove deprecated options for 1.3.0 release
* :hpx-pr:`3789` - Remove Boost filesystem compatibility header
* :hpx-pr:`3788` - Disabled even more spots that should not execute if
  networking is disabled
* :hpx-pr:`3787` - Bump minimal boost supported version to 1.61.0
* :hpx-pr:`3786` - Bump minimum required versions for 1.3.0 release
* :hpx-pr:`3785` - Explicitly set number of jobs for all ninja invocations on
  CircleCI
* :hpx-pr:`3784` - Fix leak and address sanitizer problems
* :hpx-pr:`3783` - Disabled even more spots that should not execute is
  networking is disabled
* :hpx-pr:`3782` - Cherry-picked tuple and thread_init_data fixes from #3701
* :hpx-pr:`3781` - Fix generic context coroutines after lazy stack allocation
  changes
* :hpx-pr:`3780` - Rename hello world examples
* :hpx-pr:`3776` - Sort algorithms now use the supplied chunker to determine the
  required minimal chunk size
* :hpx-pr:`3775` - Disable Boost auto-linking
* :hpx-pr:`3774` - Tag and push stable builds
* :hpx-pr:`3773` - Enable migration of polymorphic components
* :hpx-pr:`3771` - Fix link to stackoverflow in documentation
* :hpx-pr:`3770` - Replacing constexpr if in brace-serialization code
* :hpx-pr:`3769` - Fix SIGSEGV handler
* :hpx-pr:`3768` - Adding flags to scheduler allowing to control thread stealing
  and idle back-off
* :hpx-pr:`3767` - Fix help formatting in hpxrun.py
* :hpx-pr:`3765` - Fix a couple of bugs in the thread test
* :hpx-pr:`3764` - Workaround for SFINAE regression in msvc14.2
* :hpx-pr:`3762` - Prevent MSVC from prematurely instantiating things
* :hpx-pr:`3761` - Update python scripts to work with python 3
* :hpx-pr:`3760` - Fix callable vtable for GCC4.9
* :hpx-pr:`3759` - Rename ``PAGE_SIZE`` to ``PAGE_SIZE_`` because AppleClang
* :hpx-pr:`3755` - Making sure locks are not held during suspension
* :hpx-pr:`3754` - Disable more code if networking is not available/not enabled
* :hpx-pr:`3752` - Move ``util::format`` implementation to source file
* :hpx-pr:`3751` - Fixing problems with ``lcos::barrier`` and iostreams
* :hpx-pr:`3750` - Change error message to take into account ``use_guard_page``
  setting
* :hpx-pr:`3749` - Fix lifetime problem in ``run_as_hpx_thread``
* :hpx-pr:`3748` - Fixed unusable behavior of the clang code analyzer.
* :hpx-pr:`3747` - Added ``PMIX_RANK`` to the defaults of
  ``HPX_WITH_PARCELPORT_MPI_ENV``.
* :hpx-pr:`3745` - Introduced ``cache_aligned_data`` and ``cache_line_data``
  helper structure
* :hpx-pr:`3742` - Remove more unused functionality from util/logging
* :hpx-pr:`3740` - Fix includes in partitioned vector tests
* :hpx-pr:`3739` - More fixes to make sure that ``std::flush`` really flushes
  all output
* :hpx-pr:`3737` - Fix potential shutdown problems
* :hpx-pr:`3736` - Fix ``guided_pool_executor`` after dataflow changes caused
  compilation fail
* :hpx-pr:`3734` - Limiting executor
* :hpx-pr:`3732` - More constrained bound constructors
* :hpx-pr:`3730` - Attempt to fix deadlocks during component loading
* :hpx-pr:`3729` - Add latch member function ``count_up`` and reset, requested
  by hpxMP
* :hpx-pr:`3728` - Send even empty buffers on ``hpx::endl`` and ``hpx::flush``
* :hpx-pr:`3727` - Adding example demonstrating how to customize the memory
  management for a component
* :hpx-pr:`3726` - Adding support for passing command line options through the
  ``HPX_COMMANDLINE_OPTIONS`` environment variable
* :hpx-pr:`3722` - Document known broken OpenMPI builds
* :hpx-pr:`3716` - Add barrier reset function, requested by hpxMP for reusing
  barrier
* :hpx-pr:`3715` - More work on functions and vtables
* :hpx-pr:`3714` - Generate single-page HTML, PDF, manpage from documentation
* :hpx-pr:`3713` - Updating default APEX version to 2.1.2
* :hpx-pr:`3712` - Update release procedure
* :hpx-pr:`3710` - Fix the C++11 build, after #3704
* :hpx-pr:`3709` - Move some component_registry functionality to source file
* :hpx-pr:`3708` - Ignore all locks while handling assertions
* :hpx-pr:`3707` - Remove obsolete hpx runtime executable
* :hpx-pr:`3705` - Fix and simplify ``make_ready_future`` overload sets
* :hpx-pr:`3704` - Reduce use of binders
* :hpx-pr:`3703` - Ini
* :hpx-pr:`3702` - Fixing CUDA compiler errors
* :hpx-pr:`3700` - Added ``barrier::increment`` function to increase total
  number of thread
* :hpx-pr:`3697` - One more attempt to fix migration...
* :hpx-pr:`3694` - Fixing component migration
* :hpx-pr:`3693` - Print thread state when getting disallowed value in
  set_thread_state
* :hpx-pr:`3692` - Only disable ``constexpr`` with clang-cuda, not nvcc+gcc
* :hpx-pr:`3691` - Link with libsupc++ if needed for thread_local
* :hpx-pr:`3690` - Remove thousands separators in set_operations_3442 to comply
  with C++11
* :hpx-pr:`3688` - Decouple serialization from function vtables
* :hpx-pr:`3687` - Fix a couple of test failures
* :hpx-pr:`3686` - Make sure tests.unit.build are run after install on CircleCI
* :hpx-pr:`3685` - Revise quickstart CMakeLists.txt explanation
* :hpx-pr:`3684` - Provide concept emulation for Ranges-TS concepts
* :hpx-pr:`3683` - Ignore uninitialized chunks
* :hpx-pr:`3682` - Ignore uninitialized chunks. Check proper indices.
* :hpx-pr:`3680` - Ignore uninitialized chunks. Check proper range indices
* :hpx-pr:`3679` - Simplify basic action implementations
* :hpx-pr:`3678` - Making sure ``HPX_HAVE_LIBATOMIC`` is unset before checking
* :hpx-pr:`3677` - Fix generated full version number to be usable in expressions
* :hpx-pr:`3674` - Reduce functional utilities call depth
* :hpx-pr:`3672` - Change new build system to use existing macros related to
  pseudo dependencies
* :hpx-pr:`3669` - Remove indirection in ``function_ref`` when thread
  description is disabled
* :hpx-pr:`3668` - Unbreaking ``async_*cb*`` tests
* :hpx-pr:`3667` - Generate version.hpp
* :hpx-pr:`3665` - Enabling MPI parcelport for gitlab runners
* :hpx-pr:`3664` - making clang-tidy work properly again
* :hpx-pr:`3662` - Attempt to fix exception handling
* :hpx-pr:`3661` - Move ``lcos::latch`` to source file
* :hpx-pr:`3660` - Fix accidentally explicit gid_type default constructor
* :hpx-pr:`3659` - Parallel executor latch
* :hpx-pr:`3658` - Fixing execution_parameters
* :hpx-pr:`3657` - Avoid dangling references in wait_all
* :hpx-pr:`3656` - Avoiding lifetime problems with sync_put_parcel
* :hpx-pr:`3655` - Fixing nullptr dereference inside of function
* :hpx-pr:`3652` - Attempt to fix ``thread_map_type`` definition with C++11
* :hpx-pr:`3650` - Allowing for end iterator being different from begin iterator
* :hpx-pr:`3649` - Added architecture identification to cmake to be able to
  detect timestamp support
* :hpx-pr:`3645` - Enabling sanitizers on gitlab runner
* :hpx-pr:`3644` - Attempt to tackle timeouts during startup
* :hpx-pr:`3642` - Cleanup parallel partitioners
* :hpx-pr:`3640` - Dataflow now works with functions that return a reference
* :hpx-pr:`3637` - Merging the executor-enabled overloads of
  ``shared_future<>::then``
* :hpx-pr:`3633` - Replace deprecated boost endian macros
* :hpx-pr:`3632` - Add instructions on getting HPX to documentation
* :hpx-pr:`3631` - Simplify parcel creation
* :hpx-pr:`3630` - Small additions and fixes to release procedure
* :hpx-pr:`3629` - Modular pp
* :hpx-pr:`3627` - Implement ``util::function_ref``
* :hpx-pr:`3626` - Fix cancelable_action_client example
* :hpx-pr:`3625` - Added automatic serialization for simple structs (see #3034)
* :hpx-pr:`3624` - Updating the default order of priority for
  ``thread_description``
* :hpx-pr:`3621` - Update copyright year and other small formatting fixes
* :hpx-pr:`3620` - Adding support for gitlab runner
* :hpx-pr:`3619` - Store debug logs and core dumps on CircleCI
* :hpx-pr:`3618` - Various optimizations
* :hpx-pr:`3617` - Fix link to the gpg key (#2)
* :hpx-pr:`3615` - Fix unused variable warnings with networking off
* :hpx-pr:`3614` - Restructuring counter data in scheduler to reduce false
  sharing
* :hpx-pr:`3613` - Adding support for gitlab runners
* :hpx-pr:`3610` - Don't wait for ``stop_condition`` in main thread
* :hpx-pr:`3608` - Add inline keyword to ``invalid_thread_id`` definition for
  nvcc
* :hpx-pr:`3607` - Adding configuration key that allows one to explicitly add a
  directory to the component search path
* :hpx-pr:`3606` - Add nvcc to exclude constexpress since is it not supported by
  nvcc
* :hpx-pr:`3605` - Add ``inline`` to definition of checkpoint stream operators
  to fix link error
* :hpx-pr:`3604` - Use format for string formatting
* :hpx-pr:`3603` - Improve the error message for using to less ``MAX_CPU_COUNT``
* :hpx-pr:`3602` - Improve the error message for to small values of
  ``MAX_CPU_COUNT``
* :hpx-pr:`3600` - Parallel executor aggregated
* :hpx-pr:`3599` - Making sure networking is disabled for default
  one-locality-runs
* :hpx-pr:`3596` - Store thread exit functions in ``forward_list`` instead of
  ``deque`` to avoid allocations
* :hpx-pr:`3590` - Fix typo/mistake in thread queue ``cleanup_terminated``
* :hpx-pr:`3588` - Fix formatting errors in
  launching_and_configuring_hpx_applications.rst
* :hpx-pr:`3586` - Make bind propagate value category
* :hpx-pr:`3585` - Extend Cmake for building hpx as distribution packages (refs
  #3575)
* :hpx-pr:`3584` - Untangle function storage from object pointer
* :hpx-pr:`3582` - Towards Modularized HPX
* :hpx-pr:`3580` - Remove extra ``||`` in merge.hpp
* :hpx-pr:`3577` - Partially revert "Remove vtable empty flag"
* :hpx-pr:`3576` - Make sure empty startup/shutdown functions are not being used
* :hpx-pr:`3574` - Make sure ``DATAPAR`` settings are conveyed to depending
  projects
* :hpx-pr:`3573` - Make sure HPX is usable with latest released version of Vc
  (V1.4.1)
* :hpx-pr:`3572` - Adding test ensuring ticket 3565 is fixed
* :hpx-pr:`3571` - Make empty ``[unique_]function`` vtable non-dependent
* :hpx-pr:`3566` - Fix compilation with dynamic bitset for CPU masks
* :hpx-pr:`3563` - Drop ``util::[unique_]function`` target_type
* :hpx-pr:`3562` - Removing the target suffixes
* :hpx-pr:`3561` - Replace executor traits return type deduction (keep
  non-SFINAE)
* :hpx-pr:`3557` - Replace the last usages of boost::atomic
* :hpx-pr:`3556` - Replace ``boost::scoped_array`` with ``std::unique_ptr``
* :hpx-pr:`3552` - (Re)move APEX readme
* :hpx-pr:`3548` - Replace ``boost::scoped_ptr`` with ``std::unique_ptr``
* :hpx-pr:`3547` - Remove last use of Boost.Signals2
* :hpx-pr:`3544` - Post 1.2.0 version bumps
* :hpx-pr:`3543` - added Ubuntu dependency list to readme
* :hpx-pr:`3531` - Warnings, warnings...
* :hpx-pr:`3527` - Add CircleCI filter for building all tags
* :hpx-pr:`3525` - Segmented algorithms
* :hpx-pr:`3517` - Replace ``boost::regex`` with C++11 ``<regex>``
* :hpx-pr:`3514` - Cleaning up the build system
* :hpx-pr:`3505` - Fixing type attribute warning for ``transfer_action``
* :hpx-pr:`3504` - Add support for rpm packaging
* :hpx-pr:`3499` - Improving spinlock pools
* :hpx-pr:`3498` - Remove thread specific ptr
* :hpx-pr:`3486` - Fix comparison for expect_connecting_localities config entry
* :hpx-pr:`3469` - Enable (existing) code for extracting stack pointer on Power
  platform
