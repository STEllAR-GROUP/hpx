..
    Copyright (C) 2007-2022 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_8_1:

===========================
|hpx| V1.8.1 (Aug 5, 2022)
===========================

This is a bugfix release with a few minor additions and resolved problems.

General changes
===============

This patch release adds a number of small new features and fixes a handful of
problems discovered since the last release, in particular:

- A lot of work has been done to improve vectorization support for our parallel
  algorithms. HPX now supports using |eve| as a vectorization backend.
- Added a simple average power consumption performance counter.
- Added performance counters related to the use of zero-copy chunks in the
  networking layer.
- More work was done towards full compatibility with the sender/receivers
  proposal |p2300|.
- Fixing sync_wait to decay the result types
- Fixed collective operations to properly avoid overalapping consecutive
  operations on the same communicator.
- Simplified the implementation of our execution policies and added mapping
  functions between those.
- Fixed performance issues with our implementation of `small_vector`.
- Serialization now works with buffers of unsigned characters.
- Fixing dangling reference in serialization of non-default constructible types
- Fixed static linking on Windows.
- Fixed support for M1/MacOS based architectures.
- Fixed support for gentoo/musl.
- Fixed `hpx::counting_semaphore_var`.
- Properly check start and end bounds for `hpx::for_loop`
- A lot of changes and fixes to the documentation (see
  https://hpx-docs.stellar-group.org).

Breaking changes
================

- No breaking changes have been introduced.

Closed issues
=============

* :hpx-issue:`5964` - component with multiple inheritance
* :hpx-issue:`5946` - dll_dlopen.hpp: error: RTLD_DI_ORIGIN was not declared in this scope with musl libc
* :hpx-issue:`5925` - Simplify implementation of execution policies
* :hpx-issue:`5924` - {what}: mmap() failed to allocate thread stack: HPX(unhandled_exception)
* :hpx-issue:`5912` - collectives all gather hangs if rank 0 is not involved
* :hpx-issue:`5902` - MPI parcelport issue on Fugaku
* :hpx-issue:`5900` - Unable to build hello_world_distributed.cpp.
* :hpx-issue:`5892` - Problems with HPX serialization as a standalone feature. Testcase provided.
* :hpx-issue:`5886` - Segfault when serializing non default constructible class with stl containers data members
* :hpx-issue:`5832` - Distributed execution crash
* :hpx-issue:`5768` - HPX hangs on Perlmutter
* :hpx-issue:`5735` - hpx::for_loop executes without checking start and end bounds
* :hpx-issue:`5700` - HPX(serialization_error)

Closed pull requests
====================

* :hpx-pr:`5970` - Fixing component multiple inheritance
* :hpx-pr:`5969` - Fixing sync_wait to avoid dangling references
* :hpx-pr:`5963` - Fixing sync_wait to decay the result types
* :hpx-pr:`5960` - docs: added name to documentation contributors list
* :hpx-pr:`5959` - Fixing sync_wait to decay the result types
* :hpx-pr:`5954` - refactor: rename itr to correct type (`reduce`)
* :hpx-pr:`5954` - refactor: rename itr to correct type (`reduce`)
* :hpx-pr:`5953` - Fixed property handling in hierarchical_spawning
* :hpx-pr:`5951` - Fixing static linking (for Windows)
* :hpx-pr:`5947` - Fix building on musl.
* :hpx-pr:`5944` - added adaptive_static_chunk_size
* :hpx-pr:`5943` - Fix sync_wait
* :hpx-pr:`5942` - Fix doc warnings
* :hpx-pr:`5941` - Fix sync_wait
* :hpx-pr:`5940` - Protect collective operations against std::vector<bool> idiosyncrasies
* :hpx-pr:`5939` - docs: fix & improve parallel algorithms documentation 2
* :hpx-pr:`5938` - Properly implement generation support for collective operations
* :hpx-pr:`5937` - Remove leftover files from PMR based small_vector
* :hpx-pr:`5936` - Adding mapping functions between execution policies
* :hpx-pr:`5935` - Fixing serialization to work with buffers of unsigned chars
* :hpx-pr:`5934` - Attempting to fix datapar issues on CircleCI
* :hpx-pr:`5933` - Fix documentation for ranges algorithms
* :hpx-pr:`5932` - Remove mimalloc version constraint
* :hpx-pr:`5931` - docs: fix & improve parallel algorithms documentation
* :hpx-pr:`5930` - Add boost to hip builder
* :hpx-pr:`5929` - Apply fixes to M1/MacOS related stack allocation to all relevant spots
* :hpx-pr:`5928` - updated context_generic_context to accommodate arm64_arch_8/Apple architecture
* :hpx-pr:`5927` - Public derivation for counting_semaphore_var
* :hpx-pr:`5926` - Fix doxygen warnings when building documentation
* :hpx-pr:`5923` - Fixing git checkout to reflect latest version tag
* :hpx-pr:`5922` - A couple of unrelated changes in support of implementing P1673
* :hpx-pr:`5920` - [P2300] enhancements: receiver_of, sender_of improvements
* :hpx-pr:`5917` - Fixing various 'held lock while suspending' problems
* :hpx-pr:`5916` - Fix minor doxygen parsing typo
* :hpx-pr:`5915` - docs: fix broken api algo links
* :hpx-pr:`5914` - Remove CSS rules - update sphinx version
* :hpx-pr:`5911` - Removed references to hpx::vector in comments
* :hpx-pr:`5909` - Remove stuff which is defined in the header
* :hpx-pr:`5906` - Use BUILD_SHARED_LIBS correctly
* :hpx-pr:`5905` - Fix incorrect usage of generator expressions
* :hpx-pr:`5904` - Delete FindBZip2.cmake
* :hpx-pr:`5901` - Fix #5900
* :hpx-pr:`5899` - Replace PMR based version of small_vector
* :hpx-pr:`5897` - Add missing ""
* :hpx-pr:`5896` - Docs: Add serialization tutorial.
* :hpx-pr:`5895` - Update to V1.9.0 on master
* :hpx-pr:`5894` - Fix executor_with_thread_hooks example
* :hpx-pr:`5890` - Adding simple average power consumption performance counter
* :hpx-pr:`5889` - Par unseq/unseq adding
* :hpx-pr:`5888` - Support for data-parallelism for reduce, transform reduce, transform_binary_reduce algorithms
* :hpx-pr:`5887` - Fixing dangling reference in serialization of non-default constructible types
* :hpx-pr:`5879` - New performance counters related to zero-copy chunks.
