.. _cmake_variables:

**************************
 CMake Variable Reference 
**************************

.. sectionauthor:: Bryce Adelstein-Lelbach 

Entries take the following format:

identifier : type : default value : valid values
  Description of the variable's semantics.

Platform-Independent Variables
------------------------------

BOOST_FOUND_LIBRARIES : LIST
  The list of Boost libraries that CMake has found.

BOOST_INCLUDE_DIR : PATH
  The include directory for Boost.

BOOST_LIBRARIES : LIST
  The list of Boost libraries to search for.

BOOST_LIBRARY_DIR : PATH
  The directory containing the Boost libraries.

BOOST_ROOT : PATH
  The root of a Boost installation or source tree.

BOOST_SUFFIX : STRING
  The suffix to use while searching for Boost. 

BOOST_USE_MULTITHREADED : BOOL : ON
  Explicitly instruct CMake to search for multi-threaded Boost libraries.

BOOST_USE_SYSTEM : BOOL : OFF
  Explicitly instruct CMake to search for a system installation of Boost.

CMAKE_BUILD_TYPE : STRING : RELEASE : Release RelWithDebInfo MinSizeRel Debug
  Specifies optimization levels and availability of debugging information.

CMAKE_CXX_COMPILER : FILEPATH
  The C++ compiler to use to build HPX.

CMAKE_INSTALL_PREFIX : PATH : /usr/local (Linux) C:/Program Files/hpx (Windows)
  The installation prefix for HPX.

HPX_CMAKE_LOGLEVEL : STRING : WARN : Error Warn Info Debug
  The verbosity of CMake logging messages.

HPX_WARNINGS : BOOL : ON
  If true, compiler warnings are enabled.

HPX_TESTS : BOOL : ON
  If true, HPX tests are configured when CMake is run.

HPX_EXAMPLES : BOOL : ON
  If true, HPX examples are configured when CMake is run.

HPX_RUNTIME : BOOL : ON
  If true, build the hpx_runtime application. 

HPX_STACKTRACES : BOOL : ON
  If true, exceptions thrown by HPX will include stack traces.

HPX_NATIVE_TLS : BOOL : ON
  If true, OS/compiler support for TLS is used. 

HPX_CXX11 : BOOL : ON
  If true, use C++11 features if the compiler supports them.

HPX_STACK_SIZE : STRING : 0x10000 (Linux) 0x4000 (Windows)
  Default stack size for HPX-threads.

Linux-Specific Variables
------------------------

HPX_CXX_FLAG_MARCH_NATIVE : STRING : DETECT : ON OFF DETECT
  Use compiler automated tuning to improve code optimization. Decreases the
  portability of the compiled binaries.

HPX_ELF_HIDDEN_VISIBILITY : BOOL : ON
  Set the default ELF symbol visibility to hidden. Decreases compiled binary
  size.

HPX_GNU_128BIT_INTEGERS : STRING : DETECT : ON OFF DETECT
  Use GCC-style 128-bit integers. 

HPX_GNU_ALIGNED_16 : STRING : DETECT : ON OFF DETECT
  Use GCC-style __attribute__((aligned(16))).

HPX_CXX_FLAG_MCX16 : STRING : DETECT : ON OFF DETECT
  Use GCC-style support for the CMPXCHG16B instruction.

HPX_INTERNAL_CHRONO : BOOL : ON 
  Use HPX's internal version of Boost.Chrono.

HPX_MALLOC : STRING : TCMalloc : TCMalloc Jemalloc System
  The malloc allocator to use for HPX. 

HPX_PTHREAD_AFFINITY_NP : STRING : DETECT : ON OFF DETECT
  Use pthread_setaffinity_np and pthread_getaffinity_np.

HPX_RDTSC : STRING : DETECT : ON OFF DETECT
  Use the RDTSC instruction.

HPX_RDTSCP : STRING : DETECT : ON OFF DETECT
  Use the RDTSCP instruction.

HPX_SSE2 : STRING : DETECT : ON OFF DETECT
  Use SSE2 extensions.

JEMALLOC_FOUND : BOOL
  Set to true if the jemalloc library is found.

JEMALLOC_INCLUDE_DIR : PATH
  The include directory for jemalloc.

JEMALLOC_LIBRARY : FILEPATH
  Path to the jemalloc library. 

JEMALLOC_ROOT : PATH
  The root of the jemalloc installation.

JEMALLOC_USE_SYSTEM : BOOL : OFF
  Explicitly instruct CMake to search for a system installation of jemalloc.

TCMALLOC_FOUND : BOOL
  Set to true if the tcmalloc library is found.

TCMALLOC_INCLUDE_DIR : PATH
  The include directory for tcmalloc.

TCMALLOC_LIBRARY : FILEPATH
  Path to the tcmalloc library. 

TCMALLOC_ROOT : PATH
  The root of the tcmalloc installation.

TCMALLOC_USE_SYSTEM : BOOL : OFF
  Explicitly instruct CMake to search for a system installation of tcmalloc.

HDF5_FOUND : BOOL
  Set to true if the main HDF5 library is found.

HDF5_INCLUDE_DIR : PATH
  The include directory for the main HDF5 library.

HDF5_LIBRARY : FILEPATH
  Path to the main HDF5 library. 

HDF5_ROOT : PATH
  The root of the main HDF5 installation.

HDF5_USE_SYSTEM : BOOL : OFF
  Explicitly instruct CMake to search for a system installation of the main HDF5
  library.

HDF5_CPP_FOUND : BOOL
  Set to true if the HDF5 C++ library is found.

HDF5_CPP_INCLUDE_DIR : PATH : ${HDF5_INCLUDE_DIR}
  The include directory for the HDF5 C++ library.

HDF5_CPP_LIBRARY : FILEPATH
  Path to the HDF5 C++ library. 

HDF5_CPP_ROOT : PATH : ${HDF5_ROOT}
  The root of the HDF5 C++ installation.

HDF5_CPP_USE_SYSTEM : BOOL : ${HDF5_USE_SYSTEM}
  Explicitly instruct CMake to search for a system installation of the HDF5 C++
  library.

