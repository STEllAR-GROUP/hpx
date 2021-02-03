..
    Copyright (C) 2007-2018 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_0_9_9:

============================================
|hpx| V0.9.9 (Oct 31, 2014, codename Spooky)
============================================

General changes
===============

We have had over 1500 commits since the last release and we have closed over 200
tickets (bugs, feature requests, pull requests, etc.). These are by far the
largest numbers of commits and resolved issues for any of the |hpx| releases so
far. We are especially happy about the large number of people who contributed
for the first time to |hpx|.

* We completed the transition from the older (non-conforming) implementation of
  ``hpx::future`` to the new and fully conforming version by removing the old
  code and by renaming the type ``hpx::unique_future`` to ``hpx::future``. In
  order to maintain backwards compatibility with existing code which uses the
  type ``hpx::unique_future`` we support the configuration variable
  ``HPX_UNIQUE_FUTURE_ALIAS``. If this variable is set to ``ON`` while running
  cmake it will additionally define a template alias for this type.
* We rewrote and significantly changed our build system. Please have a look at
  the new (now generated) documentation here: :ref:`hpx_build_system`. Please
  revisit your build scripts to adapt to the changes. The most notable changes
  are:

   * ``HPX_NO_INSTALL`` is no longer necessary.
   * For external builds, you need to set ``HPX_DIR`` instead of ``HPX_ROOT`` as
     described here: :ref:`using_hpx_cmake`.
   * IDEs that support multiple configurations (Visual Studio and XCode) can now
     be used as intended. that means no build dir.
   * Building HPX statically (without dynamic libraries) is now supported
     (``-DHPX_STATIC_LINKING=On``).
   * Please note that many variables used to configure the build process have
     been renamed to unify the naming conventions (see the section
     :ref:`cmake_variables` for more information).
   * This also fixes a long list of issues, for more information see
     :hpx-issue:`1204`.
* We started to implement various proposals to the C++ Standardization committee
  related to parallelism and concurrency, most notably |cpp11_n4104|_ (Working
  Draft, Technical Specification for C++ Extensions for Parallelism),
  |cpp11_n4088|_ (Task Region Rev. 3), and |cpp11_n4107|_ (Working Draft,
  Technical Specification for C++ Extensions for Concurrency).
* We completely remodeled our automatic build system to run builds and unit
  tests on various systems and compilers. This allows us to find most bugs right
  as they were introduced and helps to maintain a high level of quality and
  compatibility. The newest build logs can be found at |hpx_buildbot|_.

Bug fixes (closed tickets)
==========================

Here is a list of the important tickets we closed for this release.

* :hpx-issue:`1296` - Rename make_error_future to make_exceptional_future,
  adjust to N4123
* :hpx-issue:`1295` - building issue
* :hpx-issue:`1293` - Transpose example
* :hpx-issue:`1292` - Wrong abs() function used in example
* :hpx-issue:`1291` - non-synchronized shift operators have been removed
* :hpx-issue:`1290` - RDTSCP is defined as true for Xeon Phi build
* :hpx-issue:`1289` - Fixing 1288
* :hpx-issue:`1288` - Add new performance counters
* :hpx-issue:`1287` - Hierarchy scheduler broken performance counters
* :hpx-issue:`1286` - Algorithm cleanup
* :hpx-issue:`1285` - Broken Links in Documentation
* :hpx-issue:`1284` - Uninitialized copy
* :hpx-issue:`1283` - missing boost::scoped_ptr includes
* :hpx-issue:`1282` - Update documentation of build options for schedulers
* :hpx-issue:`1281` - reset idle rate counter
* :hpx-issue:`1280` - Bug when executing on Intel MIC
* :hpx-issue:`1279` - Add improved when_all/wait_all
* :hpx-issue:`1278` - Implement improved when_all/wait_all
* :hpx-issue:`1277` - feature request: get access to argc argv and variables_map
* :hpx-issue:`1276` - Remove merging map
* :hpx-issue:`1274` - Weird (wrong) string code in papi.cpp
* :hpx-issue:`1273` - Sequential task execution policy
* :hpx-issue:`1272` - Avoid CMake name clash for Boost.Thread library
* :hpx-issue:`1271` - Updates on HPX Test Units
* :hpx-issue:`1270` - hpx/util/safe_lexical_cast.hpp is added
* :hpx-issue:`1269` - Added default value for "LIB" cmake variable
* :hpx-issue:`1268` - Memory Counters not working
* :hpx-issue:`1266` - FindHPX.cmake is not installed
* :hpx-issue:`1263` - apply_remote test takes too long
* :hpx-issue:`1262` - Chrono cleanup
* :hpx-issue:`1261` - Need make install for papi counters and this builds all
  the examples
* :hpx-issue:`1260` - Documentation of Stencil example claims
* :hpx-issue:`1259` - Avoid double-linking Boost on Windows
* :hpx-issue:`1257` - Adding additional parameter to create_thread
* :hpx-issue:`1256` - added buildbot changes to release notes
* :hpx-issue:`1255` - Cannot build MiniGhost
* :hpx-issue:`1253` - hpx::thread defects
* :hpx-issue:`1252` - HPX_PREFIX is too fragile
* :hpx-issue:`1250` - switch_to_fiber_emulation does not work properly
* :hpx-issue:`1249` - Documentation is generated under Release folder
* :hpx-issue:`1248` - Fix usage of hpx_generic_coroutine_context and get tests
  passing on powerpc
* :hpx-issue:`1247` - Dynamic linking error
* :hpx-issue:`1246` - Make cpuid.cpp C++11 compliant
* :hpx-issue:`1245` - HPX fails on startup (setting thread affinity mask)
* :hpx-issue:`1244` - HPX_WITH_RDTSC configure test fails, but should succeed
* :hpx-issue:`1243` - CTest dashboard info for CSCS CDash drop location
* :hpx-issue:`1242` - Mac fixes
* :hpx-issue:`1241` - Failure in Distributed with Boost 1.56
* :hpx-issue:`1240` - fix a race condition in examples.diskperf
* :hpx-issue:`1239` - fix wait_each in examples.diskperf
* :hpx-issue:`1238` - Fixed #1237: hpx::util::portable_binary_iarchive failed
* :hpx-issue:`1237` - hpx::util::portable_binary_iarchive faileds
* :hpx-issue:`1235` - Fixing clang warnings and errors
* :hpx-issue:`1234` - TCP runs fail: Transport endpoint is not connected
* :hpx-issue:`1233` - Making sure the correct number of threads is registered
  with AGAS
* :hpx-issue:`1232` - Fixing race in wait_xxx
* :hpx-issue:`1231` - Parallel minmax
* :hpx-issue:`1230` - Distributed run of 1d_stencil_8 uses less threads than
  spec. & sometimes gives errors
* :hpx-issue:`1229` - Unstable number of threads
* :hpx-issue:`1228` - HPX link error (cmake / MPI)
* :hpx-issue:`1226` - Warning about struct/class thread_counters
* :hpx-issue:`1225` - Adding parallel::replace etc
* :hpx-issue:`1224` - Extending dataflow to pass through non-future arguments
* :hpx-issue:`1223` - Remaining find algorithms implemented, N4071
* :hpx-issue:`1222` - Merging all the changes
* :hpx-issue:`1221` - No error output when using mpirun with hpx
* :hpx-issue:`1219` - Adding new AGAS cache performance counters
* :hpx-issue:`1216` - Fixing using futures (clients) as arguments to actions
* :hpx-issue:`1215` - Error compiling simple component
* :hpx-issue:`1214` - Stencil docs
* :hpx-issue:`1213` - Using more than a few dozen MPI processes on SuperMike
  results in a seg fault before getting to hpx_main
* :hpx-issue:`1212` - Parallel rotate
* :hpx-issue:`1211` - Direct actions cause the future's shared_state to be
  leaked
* :hpx-issue:`1210` - Refactored local::promise to be standard conformant
* :hpx-issue:`1209` - Improve command line handling
* :hpx-issue:`1208` - Adding parallel::reverse and parallel::reverse_copy
* :hpx-issue:`1207` - Add copy_backward and move_backward
* :hpx-issue:`1206` - N4071 additional algorithms implemented
* :hpx-issue:`1204` - Cmake simplification and various other minor changes
* :hpx-issue:`1203` - Implementing new launch policy for (local) async:
  ``hpx::launch::fork``.
* :hpx-issue:`1202` - Failed assertion in connection_cache.hpp
* :hpx-issue:`1201` - pkg-config doesn't add mpi link directories
* :hpx-issue:`1200` - Error when querying time performance counters
* :hpx-issue:`1199` - library path is now configurable (again)
* :hpx-issue:`1198` - Error when querying performance counters
* :hpx-issue:`1197` - tests fail with intel compiler
* :hpx-issue:`1196` - Silence several warnings
* :hpx-issue:`1195` - Rephrase initializers to work with VC++ 2012
* :hpx-issue:`1194` - Simplify parallel algorithms
* :hpx-issue:`1193` - Adding ``parallel::equal``
* :hpx-issue:`1192` - HPX(out_of_memory) on including <hpx/hpx.hpp>
* :hpx-issue:`1191` - Fixing #1189
* :hpx-issue:`1190` - Chrono cleanup
* :hpx-issue:`1189` - Deadlock .. somewhere? (probably serialization)
* :hpx-issue:`1188` - Removed ``future::get_status()``
* :hpx-issue:`1186` - Fixed FindOpenCL to find current AMD APP SDK
* :hpx-issue:`1184` - Tweaking future unwrapping
* :hpx-issue:`1183` - Extended ``parallel::reduce``
* :hpx-issue:`1182` - ``future::unwrap`` hangs for ``launch::deferred``
* :hpx-issue:`1181` - Adding ``all_of``, ``any_of``, and ``none_of`` and
  corresponding documentation
* :hpx-issue:`1180` - ``hpx::cout`` defect
* :hpx-issue:`1179` - ``hpx::async`` does not work for member function pointers
  when called on types with self-defined unary ``operator*``
* :hpx-issue:`1178` - Implemented variadic ``hpx::util::zip_iterator``
* :hpx-issue:`1177` - MPI parcelport defect
* :hpx-issue:`1176` - ``HPX_DEFINE_COMPONENT_CONST_ACTION_TPL`` does not have a
  2-argument version
* :hpx-issue:`1175` - Create util::zip_iterator working with util::tuple<>
* :hpx-issue:`1174` - Error Building HPX on linux,
  root_certificate_authority.cpp
* :hpx-issue:`1173` - hpx::cout output lost
* :hpx-issue:`1172` - HPX build error with Clang 3.4.2
* :hpx-issue:`1171` - ``CMAKE_INSTALL_PREFIX`` ignored
* :hpx-issue:`1170` - Close hpx_benchmarks repository on Github
* :hpx-issue:`1169` - Buildbot emails have syntax error in url
* :hpx-issue:`1167` - Merge partial implementation of standards proposal N3960
* :hpx-issue:`1166` - Fixed several compiler warnings
* :hpx-issue:`1165` - cmake warns: "tests.regressions.actions" does not exist
* :hpx-issue:`1164` - Want my own serialization of hpx::future
* :hpx-issue:`1162` - Segfault in hello_world example
* :hpx-issue:`1161` - Use ``HPX_ASSERT`` to aid the compiler
* :hpx-issue:`1160` - Do not put -DNDEBUG into hpx_application.pc
* :hpx-issue:`1159` - Support Clang 3.4.2
* :hpx-issue:`1158` - Fixed #1157: Rename when_n/wait_n, add
  when_xxx_n/wait_xxx_n
* :hpx-issue:`1157` - Rename when_n/wait_n, add when_xxx_n/wait_xxx_n
* :hpx-issue:`1156` - Force inlining fails
* :hpx-issue:`1155` - changed header of printout to be compatible with python
  csv module
* :hpx-issue:`1154` - Fixing iostreams
* :hpx-issue:`1153` - Standard manipulators (like std::endl) do not work with
  hpx::ostream
* :hpx-issue:`1152` - Functions revamp
* :hpx-issue:`1151` - Suppressing cmake 3.0 policy warning for CMP0026
* :hpx-issue:`1150` - Client Serialization error
* :hpx-issue:`1149` - Segfault on Stampede
* :hpx-issue:`1148` - Refactoring mini-ghost
* :hpx-issue:`1147` - N3960 copy_if and copy_n implemented and tested
* :hpx-issue:`1146` - Stencil print
* :hpx-issue:`1145` - N3960 hpx::parallel::copy implemented and tested
* :hpx-issue:`1144` - OpenMP examples 1d_stencil do not build
* :hpx-issue:`1143` - 1d_stencil OpenMP examples do not build
* :hpx-issue:`1142` - Cannot build HPX with gcc 4.6 on OS X
* :hpx-issue:`1140` - Fix OpenMP lookup, enable usage of config tests in
  external CMake projects.
* :hpx-issue:`1139` - hpx/hpx/config/compiler_specific.hpp
* :hpx-issue:`1138` - clean up pkg-config files
* :hpx-issue:`1137` - Improvements to create binary packages
* :hpx-issue:`1136` - HPX_GCC_VERSION not defined on all compilers
* :hpx-issue:`1135` - Avoiding collision between winsock2.h and windows.h
* :hpx-issue:`1134` - Making sure, that hpx::finalize can be called from any
  locality
* :hpx-issue:`1133` - 1d stencil examples
* :hpx-issue:`1131` - Refactor unique_function implementation
* :hpx-issue:`1130` - Unique function
* :hpx-issue:`1129` - Some fixes to the Build system on OS X
* :hpx-issue:`1128` - Action future args
* :hpx-issue:`1127` - Executor causes segmentation fault
* :hpx-issue:`1124` - Adding new API functions: ``register_id_with_basename``,
  ``unregister_id_with_basename``, ``find_ids_from_basename``; adding test
* :hpx-issue:`1123` - Reduce nesting of try-catch construct in
  ``encode_parcels``?
* :hpx-issue:`1122` - Client base fixes
* :hpx-issue:`1121` - Update ``hpxrun.py.in``
* :hpx-issue:`1120` - HTTS2 tests compile errors on v110 (VS2012)
* :hpx-issue:`1119` - Remove references to boost::atomic in accumulator example
* :hpx-issue:`1118` - Only build test thread_pool_executor_1114_test if
  ``HPX_LOCAL_SCHEDULER`` is set
* :hpx-issue:`1117` - local_queue_executor linker error on vc110
* :hpx-issue:`1116` - Disabled performance counter should give runtime errors,
  not invalid data
* :hpx-issue:`1115` - Compile error with Intel C++ 13.1
* :hpx-issue:`1114` - Default constructed executor is not usable
* :hpx-issue:`1113` - Fast compilation of logging causes ABI incompatibilities
  between different ``NDEBUG`` values
* :hpx-issue:`1112` - Using thread_pool_executors causes segfault
* :hpx-issue:`1111` - ``hpx::threads::get_thread_data`` always returns zero
* :hpx-issue:`1110` - Remove unnecessary null pointer checks
* :hpx-issue:`1109` - More tests adjustments
* :hpx-issue:`1108` - Clarify build rules for "libboost_atomic-mt.so"?
* :hpx-issue:`1107` - Remove unnecessary null pointer checks
* :hpx-issue:`1106` - network_storage benchmark improvements, adding legends to
  plots and tidying layout
* :hpx-issue:`1105` - Add more plot outputs and improve instructions doc
* :hpx-issue:`1104` - Complete quoting for parameters of some CMake commands
* :hpx-issue:`1103` - Work on test/scripts
* :hpx-issue:`1102` - Changed minimum requirement of window install to 2012
* :hpx-issue:`1101` - Changed minimum requirement of window install to 2012
* :hpx-issue:`1100` - Changed readme to no longer specify using MSVC 2010
  compiler
* :hpx-issue:`1099` - Error returning futures from component actions
* :hpx-issue:`1098` - Improve storage test
* :hpx-issue:`1097` - data_actions quickstart example calls missing function
  decorate_action of data_get_action
* :hpx-issue:`1096` - MPI parcelport broken with new zero copy optimization
* :hpx-issue:`1095` - Warning C4005: _WIN32_WINNT: Macro redefinition
* :hpx-issue:`1094` - Syntax error for -DHPX_UNIQUE_FUTURE_ALIAS in master
* :hpx-issue:`1093` - Syntax error for -DHPX_UNIQUE_FUTURE_ALIAS
* :hpx-issue:`1092` - Rename unique_future<> back to future<>
* :hpx-issue:`1091` - Inconsistent error message
* :hpx-issue:`1090` - On windows 8.1 the examples crashed if using more than one
  os thread
* :hpx-issue:`1089` - Components should be allowed to have their own executor
* :hpx-issue:`1088` - Add possibility to select a network interface for the
  ibverbs parcelport
* :hpx-issue:`1087` - ibverbs and ipc parcelport uses zero copy optimization
* :hpx-issue:`1083` - Make shell examples copyable in docs
* :hpx-issue:`1082` - Implement proper termination detection during shutdown
* :hpx-issue:`1081` - Implement thread_specific_ptr for hpx::threads
* :hpx-issue:`1072` - make install not working properly
* :hpx-issue:`1070` - Complete quoting for parameters of some CMake commands
* :hpx-issue:`1059` - Fix more unused variable warnings
* :hpx-issue:`1051` - Implement when_each
* :hpx-issue:`973` - Would like option to report hwloc bindings
* :hpx-issue:`970` - Bad flags for Fortran compiler
* :hpx-issue:`941` - Create a proper user level context switching class for BG/Q
* :hpx-issue:`935` - Build error with gcc 4.6 and Boost 1.54.0 on hpx trunk and
  0.9.6
* :hpx-issue:`934` - Want to build HPX without dynamic libraries
* :hpx-issue:`927` - Make hpx/lcos/reduce.hpp accept futures of id_type
* :hpx-issue:`926` - All unit tests that are run with more than one thread with
  CTest/hpx_run_test should configure hpx.os_threads
* :hpx-issue:`925` - regression_dataflow_791 needs to be brought in line with
  HPX standards
* :hpx-issue:`899` - Fix race conditions in regression tests
* :hpx-issue:`879` - Hung test leads to cascading test failure; make tests
  should support the MPI parcelport
* :hpx-issue:`865` - future<T> and friends shall work for movable only Ts
* :hpx-issue:`847` - Dynamic libraries are not installed on OS X
* :hpx-issue:`816` - First Program tutorial pull request
* :hpx-issue:`799` - Wrap lexical_cast to avoid exceptions
* :hpx-issue:`720` - broken configuration when using ccmake on Ubuntu
* :hpx-issue:`622` - ``--hpx:hpx`` and ``--hpx:debug-hpx-log`` is nonsensical
* :hpx-issue:`525` - Extend barrier LCO test to run in distributed
* :hpx-issue:`515` - Multi-destination version of hpx::apply is broken
* :hpx-issue:`509` - Push Boost.Atomic changes upstream
* :hpx-issue:`503` - Running HPX applications on Windows should not require
  setting %PATH%
* :hpx-issue:`461` - Add a compilation sanity test
* :hpx-issue:`456` - hpx_run_tests.py should log output from tests that timeout
* :hpx-issue:`454` - Investigate threadmanager performance
* :hpx-issue:`345` - Add more versatile environmental/cmake variable support to
  hpx_find_* CMake macros
* :hpx-issue:`209` - Support multiple configurations in generated build files
* :hpx-issue:`190` - hpx::cout should be a std::ostream
* :hpx-issue:`189` - iostreams component should use startup/shutdown functions
* :hpx-issue:`183` - Use Boost.ICL for correctness in AGAS
* :hpx-issue:`44` - Implement real futures

