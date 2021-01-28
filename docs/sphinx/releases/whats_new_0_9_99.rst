..
    Copyright (C) 2007-2018 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_0_9_99:

============================
|hpx| V0.9.99 (Jul 15, 2016)
============================

General changes
===============

As the version number of this release hints, we consider this release to be a
preview for the upcoming |hpx| V1.0. All of the functionalities we set out to
implement for V1.0 are in place; all of the features we wanted to have exposed
are ready. We are very happy with the stability and performance of |hpx| and we
would like to present this release to the community in order for us to gather
broad feedback before releasing V1.0. We still expect for some minor details to
change, but on the whole this release represents what we would like to have in a
V1.0.

Overall, since the last release we have had almost 1600 commits while closing
almost 400 tickets. These numbers reflect the incredible development activity we
have seen over the last couple of months. We would like to express a big 'Thank
you!' to all contributors and those who helped to make this release happen.

The most notable addition in terms of new functionality available with this
release is the full implementation of object migration (i.e. the ability to
transparently move |hpx| components to a different compute node). Additionally,
this release of |hpx| cleans up many minor issues and some API inconsistencies.

Here are some of the main highlights and changes for this release (in no
particular order):

* We have fixed a couple of issues in AGAS and the parcel layer which have
  caused hangs, segmentation faults at exit, and a slowdown of applications over
  time. Fixing those has significantly increased the overall stability and
  performance of distributed runs.
* We have started to add parallel algorithm overloads based on the C++
  Extensions for Ranges (|cpp17_n4560|_) proposal. This also includes the
  addition of projections to the existing algorithms. Please see
  :hpx-issue:`1668` for a list of algorithms which have been adapted to
  |cpp17_n4560|_.
* We have implemented index-based parallel for-loops based on a corresponding
  standardization proposal (|cpp20_p0075r1|_). Please see :hpx-issue:`2016` for
  a list of available algorithms.
* We have added implementations for more parallel algorithms as proposed for the
  upcoming C++ 17 Standard. See :hpx-issue:`1141` for an overview of which
  algorithms are available by now.
* We have started to implement a new prototypical functionality with
  |hpx_compute| which uniformly exposes some of the higher level APIs to
  heterogeneous architectures (currently CUDA). This functionality is an early
  preview and should not be considered stable. It may change considerably in the
  future.
* We have pervasively added (optional) executor arguments to all API functions
  which schedule new work. Executors are now used throughout the code base as
  the main means of executing tasks.
* Added ``hpx::make_future<R>(future<T> &&)`` allowing to convert a future of
  any type ``T`` into a future of any other type ``R``, either based on default
  conversion rules of the embedded types or using a given explicit conversion
  function.
* We finally finished the implementation of transparent migration of components
  to another locality. It is now possible to trigger a migration operation
  without 'stopping the world' for the object to migrate. |hpx| will make sure
  that no work is being performed on an object before it is migrated and that
  all subsequently scheduled work for the migrated object will be transparently
  forwarded to the new locality. Please note that the global id of the migrated
  object does not change, thus the application will not have to be changed in
  any way to support this new functionality. Please note that this feature is
  currently considered experimental. See :hpx-issue:`559` and :hpx-pr:`1966` for
  more details.
* The ``hpx::dataflow`` facility is now usable with actions. Similarly to
  ``hpx::async``, actions can be specified as an explicit template argument
  (``hpx::dataflow<Action>(target, ...)``) or as the first argument
  (``hpx::dataflow(Action(), target, ...)``). We have also enabled the use of
  distribution policies as the target for dataflow invocations. Please see
  :hpx-issue:`1265` and :hpx-pr:`1912` for more information.
* Adding overloads of ``gather_here`` and ``gather_there`` to accept the plain
  values of the data to gather (in addition to the existing overloads expecting
  futures).
* We have cleaned up and refactored large parts of the code base. This helped
  reducing compile and link times of |hpx| itself and also of applications
  depending on it. We have further decreased the dependency of |hpx| on the
  Boost libraries by replacing part of those with facilities available from the
  standard libraries.
* Wherever possible we have removed dependencies of our API on Boost by
  replacing those with the equivalent facility from the C++11 standard library.
* We have added new performance counters for parcel coalescing, file-IO, the
  AGAS cache, and overall scheduler time. Resetting performance counters has
  been overhauled and fixed.
* We have introduced a generic client type ``hpx::components::client<>`` and
  added support for using it with ``hpx::async``. This removes the necessity to
  implement specific client types for every component type without losing type
  safety. This deemphasizes the need for using the low level ``hpx::id_type``
  for referencing (possibly remote) component instances. The plan is to
  deprecate the direct use of ``hpx::id_type`` in user code in the future.
* We have added a special iterator which supports automatic prefetching of one
  or more arrays for speeding up loop-like code (see
  ``hpx::parallel::util::make_prefetcher_context()``).
* We have extended the interfaces exposed from executors (as proposed by
  |cpp11_n4406|_) to accept an arbitrary number of arguments.

Breaking changes
================

* In order to move the dataflow facility to ``namespace hpx`` we added a
  definition of ``hpx::dataflow`` which might create ambiguities in existing
  codes. The previous definition of this facility (``hpx::lcos::local::dataflow``)
  has been deprecated and is available only if the constant
  ``-DHPX_WITH_LOCAL_DATAFLOW_COMPATIBILITY=On`` to |cmake|_ is defined at
  configuration time.
  Please explicitly qualify all uses of the dataflow facility if you enable
  this compatibility setting and encounter ambiguities.
* The adaptation of the C++ Extensions for Ranges (|cpp17_n4560|_) proposal
  imposes some breaking changes related to the return types of some of the
  parallel algorithms. Please see :hpx-issue:`1668` for a list of algorithms which
  have already been adapted.
* The facility ``hpx::lcos::make_future_void()`` has been replaced by
  ``hpx::make_future<void>()``.
* We have removed support for Intel V13 and gcc 4.4.x.
* We have removed (default) support for the generic
  ``hpx::parallel::execution_poliy`` because it was removed from the Parallelism
  TS (__cpp11_n4104__) while it was being added to the upcoming C++17 Standard.
  This facility can be still enabled at configure time by specifying
  ``-DHPX_WITH_GENERIC_EXECUTION_POLICY=On`` to |cmake|.
* Uses of ``boost::shared_ptr`` and related facilities have been replaced with
  ``std::shared_ptr`` and friends. Uses of ``boost::unique_lock``,
  ``boost::lock_guard`` etc. have also been replaced by the equivalent (and
  equally named) tools available from the C++11 standard library.
* Facilities that used to expect an explicit ``boost::unique_lock`` now take an
  ``std::unique_lock``. Additionally, ``condition_variable`` no longer aliases
  ``condition_variable_any``; its interface now only works with
  ``std::unique_lock<local::mutex>``.
* Uses of ``boost::function``, ``boost::bind``, ``boost::tuple`` have been replaced
  by the corresponding facilities in |hpx| (``hpx::util::function``,
  ``hpx::util::bind``, and ``hpx::util::tuple``, respectively).

Bug fixes (closed tickets)
==========================

Here is a list of the important tickets we closed for this release.

* :hpx-pr:`2250` - change default chunker of parallel executor to static one
* :hpx-pr:`2247` - HPX on ppc64le
* :hpx-pr:`2244` - Fixing MSVC problems
* :hpx-pr:`2238` - Fixing small typos
* :hpx-pr:`2237` - Fixing small typos
* :hpx-pr:`2234` - Fix broken add test macro when extra args are passed in
* :hpx-pr:`2231` - Fixing possible race during future awaiting in serialization
* :hpx-pr:`2230` - Fix stream nvcc
* :hpx-pr:`2229` - Fixed run_as_hpx_thread
* :hpx-pr:`2228` - On prefetching_test branch : adding prefetching_iterator and
  related tests used for prefetching containers within lambda functions
* :hpx-pr:`2227` - Support for HPXCL's opencl::event
* :hpx-pr:`2226` - Preparing for release of V0.9.99
* :hpx-pr:`2225` - fix issue when compiling components with hpxcxx
* :hpx-pr:`2224` - Compute alloc fix
* :hpx-pr:`2223` - Simplify promise
* :hpx-pr:`2222` - Replace last uses of boost::function by util::function_nonser
* :hpx-pr:`2221` - Fix config tests
* :hpx-pr:`2220` - Fixing gcc 4.6 compilation issues
* :hpx-pr:`2219` - nullptr support for ``[unique_]function``
* :hpx-pr:`2218` - Introducing clang tidy
* :hpx-pr:`2216` - Replace NULL with nullptr
* :hpx-issue:`2214` - Let inspect flag use of NULL, suggest nullptr instead
* :hpx-pr:`2213` - Require support for nullptr
* :hpx-pr:`2212` - Properly find jemalloc through pkg-config
* :hpx-pr:`2211` - Disable a couple of warnings reported by Intel on Windows
* :hpx-pr:`2210` - Fixed host::block_allocator::bulk_construct
* :hpx-pr:`2209` - Started to clean up new sort algorithms, made things compile
  for sort_by_key
* :hpx-pr:`2208` - A couple of fixes that were exposed by a new sort algorithm
* :hpx-pr:`2207` - Adding missing includes in /hpx/include/serialization.hpp
* :hpx-pr:`2206` - Call package_action::get_future before package_action::apply
* :hpx-pr:`2205` - The indirect_packaged_task::operator() needs to be run on a
  HPX thread
* :hpx-pr:`2204` - Variadic executor parameters
* :hpx-pr:`2203` - Delay-initialize members of partitioned iterator
* :hpx-pr:`2202` - Added segmented fill for hpx::vector
* :hpx-issue:`2201` - Null Thread id encountered on partitioned_vector
* :hpx-pr:`2200` - Fix hangs
* :hpx-pr:`2199` - Deprecating hpx/traits.hpp
* :hpx-pr:`2198` - Making explicit inclusion of external libraries into build
* :hpx-pr:`2197` - Fix typo in QT CMakeLists
* :hpx-pr:`2196` - Fixing a gcc warning about attributes being ignored
* :hpx-pr:`2194` - Fixing partitioned_vector_spmd_foreach example
* :hpx-issue:`2193` - partitioned_vector_spmd_foreach seg faults
* :hpx-pr:`2192` - Support Boost.Thread v4
* :hpx-pr:`2191` - HPX.Compute prototype
* :hpx-pr:`2190` - Spawning operation on new thread if remaining stack space
  becomes too small
* :hpx-pr:`2189` - Adding callback taking index and future to when_each
* :hpx-pr:`2188` - Adding new example demonstrating receive_buffer
* :hpx-pr:`2187` - Mask 128-bit ints if CUDA is being used
* :hpx-pr:`2186` - Make startup & shutdown functions unique_function
* :hpx-pr:`2185` - Fixing logging output not to cause hang on shutdown
* :hpx-pr:`2184` - Allowing component clients as action return types
* :hpx-issue:`2183` - Enabling logging output causes hang on shutdown
* :hpx-issue:`2182` - 1d_stencil seg fault
* :hpx-issue:`2181` - Setting small stack size does not change default
* :hpx-pr:`2180` - Changing default bind mode to balanced
* :hpx-pr:`2179` - adding prefetching_iterator and related tests used for
  prefetching containers within lambda functions
* :hpx-pr:`2177` - Fixing 2176
* :hpx-issue:`2176` - Launch process test fails on OSX
* :hpx-pr:`2175` - Fix unbalanced config/warnings includes, add some new ones
* :hpx-pr:`2174` - Fix test categorization : regression not unit
* :hpx-issue:`2172` - Different performance results
* :hpx-issue:`2171` - "negative entry in reference count table" running
  octotiger on 32 nodes on queenbee
* :hpx-issue:`2170` - Error while compiling on Mac + boost 1.60
* :hpx-pr:`2168` - Fixing problems with is_bitwise_serializable
* :hpx-issue:`2167` - startup & shutdown function should accept unique_function
* :hpx-issue:`2166` - Simple receive_buffer example
* :hpx-pr:`2165` - Fix wait all
* :hpx-pr:`2164` - Fix wait all
* :hpx-pr:`2163` - Fix some typos in config tests
* :hpx-pr:`2162` - Improve #includes
* :hpx-pr:`2160` - Add inspect check for missing #include <list>
* :hpx-pr:`2159` - Add missing finalize call to stop test hanging
* :hpx-pr:`2158` - Algo fixes
* :hpx-pr:`2157` - Stack check
* :hpx-issue:`2156` - OSX reports stack space incorrectly (generic context
  coroutines)
* :hpx-issue:`2155` - Race condition suspected in runtime
* :hpx-pr:`2154` - Replace boost::detail::atomic_count with the new
  util::atomic_count
* :hpx-pr:`2153` - Fix stack overflow on OSX
* :hpx-pr:`2152` - Define is_bitwise_serializable as is_trivially_copyable when
  available
* :hpx-pr:`2151` - Adding missing <cstring> for std::mem* functions
* :hpx-issue:`2150` - Unable to use component clients as action return types
* :hpx-pr:`2149` - std::memmove copies bytes, use bytes*sizeof(type) when
  copying larger types
* :hpx-pr:`2146` - Adding customization point for parallel copy/move
* :hpx-pr:`2145` - Applying changes to address warnings issued by latest version
  of PVS Studio
* :hpx-issue:`2148` - hpx::parallel::copy is broken after trivially copyable
  changes
* :hpx-pr:`2144` - Some minor tweaks to compute prototype
* :hpx-pr:`2143` - Added Boost version support information over OSX platform
* :hpx-pr:`2142` - Fixing memory leak in example
* :hpx-pr:`2141` - Add missing specializations in execution policies
* :hpx-pr:`2139` - This PR fixes a few problems reported by Clang's Undefined
  Behavior sanitizer
* :hpx-pr:`2138` - Revert "Adding fedora docs"
* :hpx-pr:`2136` - Removed double semicolon
* :hpx-pr:`2135` - Add deprecated #include check for hpx_fwd.hpp
* :hpx-pr:`2134` - Resolved memory leak in stencil_8
* :hpx-pr:`2133` - Replace uses of boost pointer containers
* :hpx-pr:`2132` - Removing unused typedef
* :hpx-pr:`2131` - Add several include checks for std facilities
* :hpx-pr:`2130` - Fixing parcel compression, adding test
* :hpx-pr:`2129` - Fix invalid attribute warnings
* :hpx-issue:`2128` - hpx::init seems to segfault
* :hpx-pr:`2127` - Making executor_traits N-nary
* :hpx-pr:`2126` - GCC 4.6 fails to deduce the correct type in lambda
* :hpx-pr:`2125` - Making parcel coalescing test actually test something
* :hpx-issue:`2124` - Make a testcase for parcel compression
* :hpx-issue:`2123` - hpx/hpx/runtime/applier_fwd.hpp - Multiple defined types
* :hpx-issue:`2122` - Exception in primary_namespace::resolve_free_list
* :hpx-issue:`2121` - Possible memory leak in 1d_stencil_8
* :hpx-pr:`2120` - Fixing 2119
* :hpx-issue:`2119` - reduce_by_key compilation problems
* :hpx-issue:`2118` - Premature unwrapping of boost::ref'ed arguments
* :hpx-pr:`2117` - Added missing initializer on last constructor for
  thread_description
* :hpx-pr:`2116` - Use a lightweight bind implementation when no placeholders
  are given
* :hpx-pr:`2115` - Replace boost::shared_ptr with std::shared_ptr
* :hpx-pr:`2114` - Adding hook functions for executor_parameter_traits
  supporting timers
* :hpx-issue:`2113` - Compilation error with gcc version 4.9.3 (MacPorts gcc49
  4.9.3_0)
* :hpx-pr:`2112` - Replace uses of safe_bool with explicit operator bool
* :hpx-issue:`2111` - Compilation error on QT example
* :hpx-issue:`2110` - Compilation error when passing non-future argument to
  unwrapped continuation in dataflow
* :hpx-issue:`2109` - Warning while compiling hpx
* :hpx-issue:`2109` - Stack trace of last bug causing issues with octotiger
* :hpx-issue:`2108` - Stack trace of last bug causing issues with octotiger
* :hpx-pr:`2107` - Making sure that a missing parcel_coalescing module does not
  cause startup exceptions
* :hpx-pr:`2106` - Stop using hpx_fwd.hpp
* :hpx-issue:`2105` - coalescing plugin handler is not optional any more
* :hpx-issue:`2104` - Make executor_traits N-nary
* :hpx-issue:`2103` - Build error with octotiger and hpx commit e657426d
* :hpx-pr:`2102` - Combining thread data storage
* :hpx-pr:`2101` - Added repartition version of 1d stencil that uses any
  performance counter
* :hpx-pr:`2100` - Drop obsolete TR1 result_of protocol
* :hpx-pr:`2099` - Replace uses of boost::bind with util::bind
* :hpx-pr:`2098` - Deprecated inspect checks
* :hpx-pr:`2097` - Reduce by key, extends #1141
* :hpx-pr:`2096` - Moving local cache from external to hpx/util
* :hpx-pr:`2095` - Bump minimum required Boost to 1.50.0
* :hpx-pr:`2094` - Add include checks for several Boost utilities
* :hpx-issue:`2093` - /.../local_cache.hpp(89): error #303: explicit type is
  missing ("int" assumed)
* :hpx-pr:`2091` - Fix for Raspberry pi build
* :hpx-pr:`2090` - Fix storage size for util::function<>
* :hpx-pr:`2089` - Fix #2088
* :hpx-issue:`2088` - More verbose output from cmake configuration
* :hpx-pr:`2087` - Making sure init_globally always executes hpx_main
* :hpx-issue:`2086` - Race condition with recent HPX
* :hpx-pr:`2085` - Adding #include checker
* :hpx-pr:`2084` - Replace boost lock types with standard library ones
* :hpx-pr:`2083` - Simplify packaged task
* :hpx-pr:`2082` - Updating APEX version for testing
* :hpx-pr:`2081` - Cleanup exception headers
* :hpx-pr:`2080` - Make call_once variadic
* :hpx-issue:`2079` - With GNU C++, line 85 of hpx/config/version.hpp causes
  link failure when linking application
* :hpx-issue:`2078` - Simple test fails with _GLIBCXX_DEBUG defined
* :hpx-pr:`2077` - Instantiate board in nqueen client
* :hpx-pr:`2076` - Moving coalescing registration to TUs
* :hpx-pr:`2075` - Fixed some documentation typos
* :hpx-pr:`2074` - Adding flush-mode to message handler flush
* :hpx-pr:`2073` - Fixing performance regression introduced lately
* :hpx-pr:`2072` - Refactor local::condition_variable
* :hpx-pr:`2071` - Timer based on boost::asio::deadline_timer
* :hpx-pr:`2070` - Refactor tuple based functionality
* :hpx-pr:`2069` - Fixed typos
* :hpx-issue:`2068` - Seg fault with octotiger
* :hpx-pr:`2067` - Algorithm cleanup
* :hpx-pr:`2066` - Split credit fixes
* :hpx-pr:`2065` - Rename HPX_MOVABLE_BUT_NOT_COPYABLE to HPX_MOVABLE_ONLY
* :hpx-pr:`2064` - Fixed some typos in docs
* :hpx-pr:`2063` - Adding example demonstrating template components
* :hpx-issue:`2062` - Support component templates
* :hpx-pr:`2061` - Replace some uses of lexical_cast<string> with C++11
  std::to_string
* :hpx-pr:`2060` - Replace uses of boost::noncopyable with HPX_NON_COPYABLE
* :hpx-pr:`2059` - Adding missing for_loop algorithms
* :hpx-pr:`2058` - Move several definitions to more appropriate headers
* :hpx-pr:`2057` - Simplify assert_owns_lock and ignore_while_checking
* :hpx-pr:`2056` - Replacing std::result_of with util::result_of
* :hpx-pr:`2055` - Fix process launching/connecting back
* :hpx-pr:`2054` - Add a forwarding coroutine header
* :hpx-pr:`2053` - Replace uses of boost::unordered_map with std::unordered_map
* :hpx-pr:`2052` - Rewrite tuple unwrap
* :hpx-pr:`2050` - Replace uses of BOOST_SCOPED_ENUM with C++11 scoped enums
* :hpx-pr:`2049` - Attempt to narrow down split_credit problem
* :hpx-pr:`2048` - Fixing gcc startup hangs
* :hpx-pr:`2047` - Fixing when_xxx and wait_xxx for MSVC12
* :hpx-pr:`2046` - adding persistent_auto_chunk_size and related tests for
  for_each
* :hpx-pr:`2045` - Fixing HPX_HAVE_THREAD_BACKTRACE_DEPTH build time
  configuration
* :hpx-pr:`2044` - Adding missing service executor types
* :hpx-pr:`2043` - Removing ambiguous definitions for is_future_range and
  future_range_traits
* :hpx-pr:`2042` - Clarify that HPX builds can use (much) more than 2GB per
  process
* :hpx-pr:`2041` - Changing future_iterator_traits to support pointers
* :hpx-issue:`2040` - Improve documentation memory usage warning?
* :hpx-pr:`2039` - Coroutine cleanup
* :hpx-pr:`2038` - Fix cmake policy CMP0042 warning MACOSX_RPATH
* :hpx-pr:`2037` - Avoid redundant specialization of [unique_]function_nonser
* :hpx-pr:`2036` - nvcc dies with an internal error upon pushing/popping
  warnings inside templates
* :hpx-issue:`2035` - Use a less restrictive iterator definition in
  hpx::lcos::detail::future_iterator_traits
* :hpx-pr:`2034` - Fixing compilation error with thread queue wait time
  performance counter
* :hpx-issue:`2033` - Compilation error when compiling with thread queue
  waittime performance counter
* :hpx-issue:`2032` - Ambiguous template instantiation for is_future_range and
  future_range_traits.
* :hpx-pr:`2031` - Don't restart timer on every incoming parcel
* :hpx-pr:`2030` - Unify handling of execution policies in parallel algorithms
* :hpx-pr:`2029` - Make pkg-config .pc files use .dylib on OSX
* :hpx-pr:`2028` - Adding process component
* :hpx-pr:`2027` - Making check for compiler compatibility independent on
  compiler path
* :hpx-pr:`2025` - Fixing inspect tool
* :hpx-pr:`2024` - Intel13 removal
* :hpx-pr:`2023` - Fix errors related to older boost versions and parameter pack
  expansions in lambdas
* :hpx-issue:`2022` - gmake fail: "No rule to make target
  /usr/lib46/libboost_context-mt.so"
* :hpx-pr:`2021` - Added Sudoku example
* :hpx-issue:`2020` - Make errors related to init_globally.cpp example while
  building HPX out of the box
* :hpx-pr:`2019` - Fixed some compilation and cmake errors encountered in nqueen
  example
* :hpx-pr:`2018` - For loop algorithms
* :hpx-pr:`2017` - Non-recursive at_index implementation
* :hpx-issue:`2016` - Add index-based for-loops
* :hpx-issue:`2015` - Change default bind-mode to balanced
* :hpx-pr:`2014` - Fixed dataflow if invoked action returns a future
* :hpx-pr:`2013` - Fixing compilation issues with external example
* :hpx-pr:`2012` - Added Sierpinski Triangle example
* :hpx-issue:`2011` - Compilation error while running sample
  hello_world_component code
* :hpx-pr:`2010` - Segmented move implemented for hpx::vector
* :hpx-issue:`2009` - pkg-config order incorrect on 14.04 / GCC 4.8
* :hpx-issue:`2008` - Compilation error in dataflow of action returning a future
* :hpx-pr:`2007` - Adding new performance counter exposing overall scheduler
  time
* :hpx-pr:`2006` - Function includes
* :hpx-pr:`2005` - Adding an example demonstrating how to initialize HPX from a
  global object
* :hpx-pr:`2004` - Fixing 2000
* :hpx-pr:`2003` - Adding generation parameter to gather to enable using it more
  than once
* :hpx-pr:`2002` - Turn on position independent code to solve link problem with
  hpx_init
* :hpx-issue:`2001` - Gathering more than once segfaults
* :hpx-issue:`2000` - Undefined reference to hpx::assertion_failed
* :hpx-issue:`1999` - Seg fault in
  hpx::lcos::base_lco_with_value<*>::set_value_nonvirt() when running octo-tiger
* :hpx-pr:`1998` - Detect unknown command line options
* :hpx-pr:`1997` - Extending thread description
* :hpx-pr:`1996` - Adding natvis files to solution (MSVC only)
* :hpx-issue:`1995` - Command line handling does not produce error
* :hpx-pr:`1994` - Possible missing include in test_utils.hpp
* :hpx-pr:`1993` - Add missing LANGUAGES tag to a
  hpx_add_compile_flag_if_available() call in CMakeLists.txt
* :hpx-pr:`1992` - Fixing shared_executor_test
* :hpx-pr:`1991` - Making sure the winsock library is properly initialized
* :hpx-pr:`1990` - Fixing bind_test placeholder ambiguity coming from boost-1.60
* :hpx-pr:`1989` - Performance tuning
* :hpx-pr:`1987` - Make configurable size of internal storage in util::function
* :hpx-pr:`1986` - AGAS Refactoring+1753 Cache mods
* :hpx-pr:`1985` - Adding missing task_block::run() overload taking an executor
* :hpx-pr:`1984` - Adding an optimized LRU Cache implementation (for AGAS)
* :hpx-pr:`1983` - Avoid invoking migration table look up for all objects
* :hpx-pr:`1981` - Replacing uintptr_t (which is not defined everywhere) with
  std::size_t
* :hpx-pr:`1980` - Optimizing LCO continuations
* :hpx-pr:`1979` - Fixing Cori
* :hpx-pr:`1978` - Fix test check that got broken in hasty fix to memory
  overflow
* :hpx-pr:`1977` - Refactor action traits
* :hpx-pr:`1976` - Fixes typo in README.rst
* :hpx-pr:`1975` - Reduce size of benchmark timing arrays to fix test failures
* :hpx-pr:`1974` - Add action to update data owned by the partitioned_vector
  component
* :hpx-pr:`1972` - Adding partitioned_vector SPMD example
* :hpx-pr:`1971` - Fixing 1965
* :hpx-pr:`1970` - Papi fixes
* :hpx-pr:`1969` - Fixing continuation recursions to not depend on fixed amount
  of recursions
* :hpx-pr:`1968` - More segmented algorithms
* :hpx-issue:`1967` - Simplify component implementations
* :hpx-pr:`1966` - Migrate components
* :hpx-issue:`1964` - fatal error: 'boost/lockfree/detail/branch_hints.hpp' file
  not found
* :hpx-issue:`1962` - parallel:copy_if has race condition when used on in place
  arrays
* :hpx-pr:`1963` - Fixing Static Parcelport initialization
* :hpx-pr:`1961` - Fix function target
* :hpx-issue:`1960` - Papi counters don't reset
* :hpx-pr:`1959` - Fixing 1958
* :hpx-issue:`1958` - inclusive_scan gives incorrect results with
  non-commutative operator
* :hpx-pr:`1957` - Fixing #1950
* :hpx-pr:`1956` - Sort by key example
* :hpx-pr:`1955` - Adding regression test for #1946: Hang in wait_all() in
  distributed run
* :hpx-issue:`1954` - HPX releases should not use -Werror
* :hpx-pr:`1953` - Adding performance analysis for AGAS cache
* :hpx-pr:`1952` - Adapting test for explicit variadics to fail for gcc 4.6
* :hpx-pr:`1951` - Fixing memory leak
* :hpx-issue:`1950` - Simplify external builds
* :hpx-pr:`1949` - Fixing yet another lock that is being held during suspension
* :hpx-pr:`1948` - Fixed container algorithms for Intel
* :hpx-pr:`1947` - Adding workaround for tagged_tuple
* :hpx-issue:`1946` - Hang in wait_all() in distributed run
* :hpx-pr:`1945` - Fixed container algorithm tests
* :hpx-issue:`1944` - assertion 'p.destination_locality() ==
  hpx::get_locality()' failed
* :hpx-pr:`1943` - Fix a couple of compile errors with clang
* :hpx-pr:`1942` - Making parcel coalescing functional
* :hpx-issue:`1941` - Re-enable parcel coalescing
* :hpx-pr:`1940` - Touching up make_future
* :hpx-pr:`1939` - Fixing problems in over-subscription management in the
  resource manager
* :hpx-pr:`1938` - Removing use of unified Boost.Thread header
* :hpx-pr:`1937` - Cleaning up the use of Boost.Accumulator headers
* :hpx-pr:`1936` - Making sure interval timer is started for aggregating
  performance counters
* :hpx-pr:`1935` - Tagged results
* :hpx-pr:`1934` - Fix remote async with deferred launch policy
* :hpx-issue:`1933` - Floating point exception in
  ``statistics_counter<boost::accumulators::tag::mean>::get_counter_value``
* :hpx-pr:`1932` - Removing superfluous includes of
  boost/lockfree/detail/branch_hints.hpp
* :hpx-pr:`1931` - fix compilation with clang 3.8.0
* :hpx-issue:`1930` - Missing online documentation for HPX 0.9.11
* :hpx-pr:`1929` - LWG2485: get() should be overloaded for const tuple&&
* :hpx-pr:`1928` - Revert "Using ninja for circle-ci builds"
* :hpx-pr:`1927` - Using ninja for circle-ci builds
* :hpx-pr:`1926` - Fixing serialization of std::array
* :hpx-issue:`1925` - Issues with static HPX libraries
* :hpx-issue:`1924` - Performance degrading over time
* :hpx-issue:`1923` - serialization of std::array appears broken in latest
  commit
* :hpx-pr:`1922` - Container algorithms
* :hpx-pr:`1921` - Tons of smaller quality improvements
* :hpx-issue:`1920` - Seg fault in hpx::serialization::output_archive::add_gid
  when running octotiger
* :hpx-issue:`1919` - Intel 15 compiler bug preventing HPX build
* :hpx-pr:`1918` - Address sanitizer fixes
* :hpx-pr:`1917` - Fixing compilation problems of parallel::sort with Intel
  compilers
* :hpx-pr:`1916` - Making sure code compiles if HPX_WITH_HWLOC=Off
* :hpx-issue:`1915` - max_cores undefined if HPX_WITH_HWLOC=Off
* :hpx-pr:`1913` - Add utility member functions for partitioned_vector
* :hpx-pr:`1912` - Adding support for invoking actions to dataflow
* :hpx-pr:`1911` - Adding first batch of container algorithms
* :hpx-pr:`1910` - Keep cmake_module_path
* :hpx-pr:`1909` - Fix mpirun with pbs
* :hpx-pr:`1908` - Changing parallel::sort to return the last iterator as
  proposed by N4560
* :hpx-pr:`1907` - Adding a minimum version for Open MPI
* :hpx-pr:`1906` - Updates to the Release Procedure
* :hpx-pr:`1905` - Fixing #1903
* :hpx-pr:`1904` - Making sure std containers are cleared before serialization
  loads data
* :hpx-issue:`1903` - When running octotiger, I get: assertion
  ``'(*new_gids_)[gid].size() == 1' failed: HPX(assertion_failure)``
* :hpx-issue:`1902` - Immediate crash when running hpx/octotiger with
  _GLIBCXX_DEBUG defined.
* :hpx-pr:`1901` - Making non-serializable classes non-serializable
* :hpx-issue:`1900` - Two possible issues with std::list serialization
* :hpx-pr:`1899` - Fixing a problem with credit splitting as revealed by #1898
* :hpx-issue:`1898` - Accessing component from locality where it was not created
  segfaults
* :hpx-pr:`1897` - Changing parallel::sort to return the last iterator as
  proposed by N4560
* :hpx-issue:`1896` - version 1.0?
* :hpx-issue:`1895` - Warning comment on numa_allocator is not very clear
* :hpx-pr:`1894` - Add support for compilers that have thread_local
* :hpx-pr:`1893` - Fixing 1890
* :hpx-pr:`1892` - Adds typed future_type for executor_traits
* :hpx-pr:`1891` - Fix wording in certain parallel algorithm docs
* :hpx-issue:`1890` - Invoking papi counters give segfault
* :hpx-pr:`1889` - Fixing problems as reported by clang-check
* :hpx-pr:`1888` - WIP parallel is_heap
* :hpx-pr:`1887` - Fixed resetting performance counters related to idle-rate,
  etc
* :hpx-issue:`1886` - Run hpx with qsub does not work
* :hpx-pr:`1885` - Warning cleaning pass
* :hpx-pr:`1884` - Add missing parallel algorithm header
* :hpx-pr:`1883` - Add feature test for thread_local on Clang for TLS
* :hpx-pr:`1882` - Fix some redundant qualifiers
* :hpx-issue:`1881` - Unable to compile Octotiger using HPX and Intel MPI on
  SuperMIC
* :hpx-issue:`1880` - clang with libc++ on Linux needs TLS case
* :hpx-pr:`1879` - Doc fixes for #1868
* :hpx-pr:`1878` - Simplify functions
* :hpx-pr:`1877` - Removing most usage of Boost.Config
* :hpx-pr:`1876` - Add missing parallel algorithms to algorithm.hpp
* :hpx-pr:`1875` - Simplify callables
* :hpx-pr:`1874` - Address long standing FIXME on using ``std::unique_ptr`` with
  incomplete types
* :hpx-pr:`1873` - Fixing 1871
* :hpx-pr:`1872` - Making sure PBS environment uses specified node list even if
  no PBS_NODEFILE env is available
* :hpx-issue:`1871` - Fortran checks should be optional
* :hpx-pr:`1870` - Touch local::mutex
* :hpx-pr:`1869` - Documentation refactoring based off #1868
* :hpx-pr:`1867` - Embrace static_assert
* :hpx-pr:`1866` - Fix #1803 with documentation refactoring
* :hpx-pr:`1865` - Setting OUTPUT_NAME as target properties
* :hpx-pr:`1863` - Use SYSTEM for boost includes
* :hpx-pr:`1862` - Minor cleanups
* :hpx-pr:`1861` - Minor Corrections for Release
* :hpx-pr:`1860` - Fixing hpx gdb script
* :hpx-issue:`1859` - reset_active_counters resets times and thread counts
  before some of the counters are evaluated
* :hpx-pr:`1858` - Release V0.9.11
* :hpx-pr:`1857` - removing diskperf example from 9.11 release
* :hpx-pr:`1856` - fix return in packaged_task_base::reset()
* :hpx-issue:`1842` - Install error: file INSTALL cannot find
  libhpx_parcel_coalescing.so.0.9.11
* :hpx-pr:`1839` - Adding fedora docs
* :hpx-pr:`1824` - Changing version on master to V0.9.12
* :hpx-pr:`1818` - Fixing #1748
* :hpx-issue:`1815` - seg fault in AGAS
* :hpx-issue:`1803` - wait_all documentation
* :hpx-issue:`1796` - Outdated documentation to be revised
* :hpx-issue:`1759` - glibc munmap_chunk or free(): invalid pointer on SuperMIC
* :hpx-issue:`1753` - HPX performance degrades with time since execution begins
* :hpx-issue:`1748` - All public HPX headers need to be self contained
* :hpx-pr:`1719` - How to build HPX with Visual Studio
* :hpx-issue:`1684` - Race condition when using --hpx:connect?
* :hpx-pr:`1658` - Add serialization for std::set (as there is for std::vector
  and std::map)
* :hpx-pr:`1641` - Generic client
* :hpx-issue:`1632` - heartbeat example fails on separate nodes
* :hpx-pr:`1603` - Adds preferred namespace check to inspect tool
* :hpx-issue:`1559` - Extend inspect tool
* :hpx-issue:`1523` - Remote async with deferred launch policy never executes
* :hpx-issue:`1472` - Serialization issues
* :hpx-issue:`1457` - Implement N4392: C++ Latches and Barriers
* :hpx-pr:`1444` - Enabling usage of moveonly types for component construction
* :hpx-issue:`1407` - The Intel 13 compiler has failing unit tests
* :hpx-issue:`1405` - Allow component constructors to take movable only types
* :hpx-issue:`1265` - Enable dataflow() to be usable with actions
* :hpx-issue:`1236` - NUMA aware allocators
* :hpx-issue:`802` - Fix Broken Examples
* :hpx-issue:`559` - Add hpx::migrate facility
* :hpx-issue:`449` - Make actions with template arguments usable and add
  documentation
* :hpx-issue:`279` - Refactor addressing_service into a base class and two
  derived classes
* :hpx-issue:`224` - Changing thread state metadata is not thread safe
* :hpx-issue:`55` - Uniform syntax for enums should be implemented

.. Proofread by:
   Adrian Serio 6-28-16
   Patricia Grubel 3-20-15
