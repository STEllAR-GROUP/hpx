# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least CMake V2.6.4
cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

################################################################################
# C++ overrides 
################################################################################
set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
    "${CMAKE_SOURCE_DIR}/cmake/HPX_CXXOverrides.cmake")

################################################################################
# project metadata
################################################################################
project(hpx CXX)

set(HPX_MAJOR_VERSION 0)
set(HPX_MINOR_VERSION 6)
set(HPX_PATCH_LEVEL   0)
set(HPX_VERSION "${HPX_MAJOR_VERSION}.${HPX_MINOR_VERSION}.${HPX_PATCH_LEVEL}")
set(HPX_SOVERSION ${HPX_MAJOR_VERSION})

################################################################################
# Python detection (part 1)
################################################################################
include(FindPythonInterp)

################################################################################
# Fortran detection (part 1)
################################################################################
include(CMakeDetermineFortranCompiler)

################################################################################
# cmake configuration
################################################################################
set(CMAKE_MODULE_PATH ${hpx_SOURCE_DIR}/cmake)

# include additional macro definitions
include(HPX_Utils)

include(HPX_Distclean)

hpx_force_out_of_tree_build("This project requires an out-of-source-tree build. See INSTALL.rst. Clean your CMake cache and CMakeFiles if this message persists.")

# be pedantic and annoying by default
if(NOT HPX_CMAKE_LOGLEVEL)
  set(HPX_CMAKE_LOGLEVEL "WARN")
endif()

################################################################################
# Python detection (part 2)
################################################################################
if(PYTHONINTERP_FOUND)
  hpx_info("python" "Found a Python interpreter")
else()
  hpx_warn("python" "Couldn't find a Python interpreter")
endif()

# SSH
hpx_check_for_python_paramiko(HPX_PYTHON_PARAMIKO
    DEFINITIONS HPX_HAVE_PYTHON_PARAMIKO) 

# Command-line parsing
hpx_check_for_python_optparse(HPX_PYTHON_OPTPARSE
    DEFINITIONS HPX_HAVE_PYTHON_OPTPARSE)

# Async networking
hpx_check_for_python_threading(HPX_PYTHON_THREADING
    DEFINITIONS HPX_HAVE_PYTHON_THREADING)

# RAII
hpx_check_for_python_with_statement(HPX_PYTHON_WITH_STATEMENT
    DEFINITIONS HPX_HAVE_PYTHON_WITH_STATEMENT)

# Process control
hpx_check_for_python_subprocess(HPX_PYTHON_SUBPROCESS
    DEFINITIONS HPX_HAVE_PYTHON_SUBPROCESS)

################################################################################
# Fortran detection (part 2)
################################################################################
if(CMAKE_Fortran_COMPILER)
  hpx_info("fortran" "Found a Fortran compiler")
  enable_language(Fortran)
else()
  hpx_warn("fortran" "Couldn't find a Fortran compiler")
endif()

################################################################################
# environment detection 
################################################################################
# FIXME: broken for MSVC
if(NOT MSVC)
  execute_process(COMMAND "${hpx_SOURCE_DIR}/python/scripts/hpx_environment.py" "${CMAKE_CXX_COMPILER}"
                  OUTPUT_VARIABLE build_environment
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(build_environment)
    set(BUILDNAME "${build_environment}" CACHE INTERNAL "A string describing the build environment.")
    hpx_info("environment" "Build environment is ${BUILDNAME}.")
  else()
    hpx_warn("environment" "Couldn't determine build environment (install python).") 
  endif()
endif()

################################################################################
# Boost configuration
################################################################################
# Boost.Chrono is in the Boost trunk, but has not been in a Boost release yet

option(HPX_INTERNAL_CHRONO "Use HPX's internal version of Boost.Chrono (default: ON)" ON)

# this cmake module will snag the Boost version we'll be using (which we need
# to know to specify the Boost libraries that we want to look for).
find_package(HPX_BoostVersion)

if(NOT BOOST_VERSION_FOUND)
  hpx_error("boost" "Failed to locate Boost.")
endif()

if(NOT HPX_INTERNAL_CHRONO OR ${BOOST_MINOR_VERSION} GREATER 46)
  set(BOOST_LIBRARIES chrono
                      date_time
                      filesystem
                      program_options
                      regex
                      serialization
                      system
                      thread)
else()
  set(BOOST_LIBRARIES date_time
                      filesystem
                      program_options
                      regex
                      serialization
                      system
                      thread)
  add_definitions(-DHPX_INTERNAL_CHRONO)
  add_definitions(-DBOOST_CHRONO_NO_LIB)
  include_directories(${hpx_SOURCE_DIR}/external/chrono)
endif()

# We have a patched version of FindBoost loosely based on the one that Kitware ships
find_package(HPX_Boost)

include_directories(${BOOST_INCLUDE_DIR})
link_directories(${BOOST_LIBRARY_DIR})

# Boost preprocessor definitions
add_definitions(-DBOOST_PARAMETER_MAX_ARITY=7)
add_definitions(-DBOOST_COROUTINE_USE_ATOMIC_COUNT) 
add_definitions(-DBOOST_COROUTINE_ARG_MAX=2)
add_definitions(-DBOOST_LOG_NO_TSS)
add_definitions(-DBOOST_LOG_NO_TS)
add_definitions(-DBOOST_BIGINT_HAS_NATIVE_INT64)

################################################################################
# search path configuration
################################################################################
include_directories(${hpx_SOURCE_DIR})
include_directories(${hpx_SOURCE_DIR}/external/move)
include_directories(${hpx_SOURCE_DIR}/external/atomic)
include_directories(${hpx_SOURCE_DIR}/external/bigint)
include_directories(${hpx_SOURCE_DIR}/external/cache)
include_directories(${hpx_SOURCE_DIR}/external/coroutine)
include_directories(${hpx_SOURCE_DIR}/external/endian)
include_directories(${hpx_SOURCE_DIR}/external/logging)
include_directories(${hpx_SOURCE_DIR}/external/lockfree)
include_directories(${hpx_SOURCE_DIR}/external/plugin)
include_directories(${hpx_SOURCE_DIR}/external/backtrace)
link_directories(${CMAKE_BINARY_DIR}/lib)
link_directories(${CMAKE_BINARY_DIR}/lib/hpx)

################################################################################
# Compiler configuration code
################################################################################
include(HPX_GCCVersion)
include(HPX_CompilerFlags)

################################################################################
# SVN revision detection 
################################################################################
include(HPX_SVNRevision)

################################################################################
# installation configuration
################################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

# for backwards compatibility
if(CMAKE_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_PREFIX}")
endif()

if(NOT CMAKE_INSTALL_PREFIX)
  if(UNIX)
    set(CMAKE_INSTALL_PREFIX "/opt/hpx" CACHE PATH "Prefix prepended to install directories.")
  else()
    set(CMAKE_INSTALL_PREFIX "C:/Program Files/hpx" CACHE PATH "Prefix prepended to install directories.")
  endif()
endif()

set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "Where to install ${PROJECT_NAME} (default: /opt/hpx for POSIX, C:/Program Files/hpx for Windows)." FORCE)

hpx_info("install" "Install root is ${CMAKE_INSTALL_PREFIX}.")

add_definitions("-DHPX_PREFIX=\"${CMAKE_INSTALL_PREFIX}\"")

if(NOT MSVC)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

################################################################################
# RPATH configuration
################################################################################
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib/hpx:${CMAKE_BINARY_DIR}/lib:${CMAKE_BINARY_DIR}/lib/hpx")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

################################################################################
# AGAS configuration
################################################################################
set(HPX_AGAS_VERSION "2" CACHE STRING "AGAS subsystem version")

if("${HPX_AGAS_VERSION}" STREQUAL "2")
  add_definitions(-DHPX_AGAS_VERSION=0x20)
else()
  add_definitions(-DHPX_AGAS_VERSION=0x10)
endif()

################################################################################
# LISP bindings 
################################################################################
find_package(HPX_Phxpr)

################################################################################
# SDF output 
################################################################################
find_package(HPX_RNPL)

################################################################################
# HDF5 output 
################################################################################
find_package(HPX_HDF5)

################################################################################
# Arbitrary precision math 
################################################################################
find_package(HPX_GMP)

#if(GMP_FOUND)
#  add_definitions(-DBOOST_BIGINT_HAS_GMP_SUPPORT)
#endif()

################################################################################
# Warning configuration
################################################################################
option(HPX_WARNINGS "Enable compiler warnings (default: ON)" ON)

################################################################################
# Backtrace configuration 
################################################################################
option(HPX_STACKTRACES "Attach backtraces to HPX exceptions (default: ON)" ON)

if(HPX_STACKTRACES)
  hpx_info("stacktraces" "Stack traces are enabled.")
  add_definitions(-DHPX_STACKTRACES=1)
endif()

################################################################################
# Windows specific configuration 
################################################################################
if(MSVC)
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DDEBUG)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    add_definitions(-DNDEBUG)
    add_definitions(-DBOOST_DISABLE_ASSERTS)
  endif()

  set(CMAKE_DEBUG_POSTFIX "d")

  # we auto-link Boost on Windows, so don't specify boost libraries explicitly
  set(BOOST_FOUND_LIBRARIES "")

  add_definitions(-D_WINDOWS)
  add_definitions(-DBOOST_USE_WINDOWS_H)
  add_definitions(-D_WIN32_WINNT=0x0600)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_DEPRECATE)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
  add_definitions(-D_CRT_NONSTDC_NO_WARNINGS)
  add_definitions(-DBOOST_SERIALIZATION_DYN_LINK)

  # suppress certain warnings (TODO: Document what warnings these disable)
  add_definitions(-wd4251 -wd4231 -wd4275 -wd4660 -wd4094 -wd4267 -wd4180 -wd4244)

  if(CMAKE_CL_64)
    add_definitions(-DBOOST_COROUTINE_USE_FIBERS)
  endif()

  # multiproccessor build
  add_definitions(-MP) 

  # TODO: implement
  #hpx_check_for_msvc_128bit_interlocked(hpx_HAVE_MSVC_128BIT_INTERLOCKED
  #  DEFINITIONS HPX_HAVE_CMPXCHG16B
  #              BOOST_ATOMIC_HAVE_MSVC_128BIT_INTERLOCKED # for the msvc code
  #              BOOST_ATOMIC_HAVE_CMPXCHG16B)             # for the fallback 

################################################################################
# POSIX specific configuration 
################################################################################
else()
  if(HPX_STACKTRACES)
    hpx_use_flag_if_available(rdynamic)
  endif()

  # clear CMake defaults
  set(CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "Debug flags (C++)" FORCE)
  set(CMAKE_CXX_FLAGS_MINSIZEREL "" CACHE STRING "MinSizeRel flags (C++)" FORCE)
  set(CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "Release flags (C++)" FORCE)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO ""  CACHE STRING "RelWithDebInfo flags (C++)" FORCE)

  # non-standard C++ extensions that we forbid
  hpx_use_flag_if_available(Werror=vla NAME Werror_vla)

  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DDEBUG)
    hpx_use_flag_if_available(g)
    hpx_use_flag_if_available(fno-omit-frame-pointer)
    hpx_use_flag_if_available(fno-inline)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    add_definitions(-DDEBUG)
    hpx_use_flag_if_available(g)
    hpx_use_flag_if_available(O3)
    hpx_use_flag_if_available(fno-omit-frame-pointer)
    hpx_use_flag_if_available(fno-inline)
    hpx_use_flag_if_available(fno-optimize-sibling-calls)
    hpx_use_flag_if_available(fno-web)
    hpx_use_flag_if_available(fno-rename-registers)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    add_definitions(-DNDEBUG)
    hpx_use_flag_if_available(Os)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    add_definitions(-DNDEBUG)
    add_definitions(-DBOOST_DISABLE_ASSERTS)
    hpx_use_flag_if_available(O3)
  else()
    hpx_error("build_type" "\"${CMAKE_BUILD_TYPE}\" is not a valid build type.")
  endif()

  if(HPX_WARNINGS)
    hpx_use_flag_if_available(Wall)
    hpx_use_flag_if_available(Wno-strict-aliasing)
    hpx_use_flag_if_available(Wsign-promo)
  endif()

  ##############################################################################
  # GNU specific configuration
  ##############################################################################
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    hpx_info("gcc_config" "Compiler reports compatibility with GCC version ${GCC_VERSION_STR}")

    # Show the flags that toggle each warning
    hpx_use_flag_if_available(fdiagnostics-show-option)

    # I'm aware that __sync_fetch_and_nand changed semantics
    hpx_use_flag_if_gcc_version(Wno-sync-nand 040400)

    if(0404 GREATER ${GCC_VERSION} OR "${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
      hpx_warn("gcc_config" "HPX will perform poorly with GCC ${GCC_VERSION_STR}. Please use GCC 4.4.x.")
    else()
      option(HPX_ELF_HIDDEN_VISIBILITY
        "Use -fvisibility=hidden for Release, MinSizeRel builds (default: ON)" ON)

      if(HPX_ELF_HIDDEN_VISIBILITY)
        if("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel" OR
           "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
          add_definitions(-DHPX_ELF_HIDDEN_VISIBILITY)
          add_definitions(-DBOOST_COROUTINE_GCC_HAVE_VISIBILITY)
          add_definitions(-DBOOST_PLUGIN_GCC_HAVE_VISIBILITY)
          hpx_append_flag(-fvisibility=hidden)
        endif()
      endif()
    endif()
  ##############################################################################
  # Intel specific configuration
  ##############################################################################
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # warning #191: type qualifier is meaningless on cast type
    add_definitions(-diag-disable 191) 
    
    # warning #279: controlling expression is constant 
    add_definitions(-diag-disable 279) 
    
    # warning #68: integer conversion resulted in a change of sign 
    add_definitions(-diag-disable 68) 
    
    # warning #858: type qualifier on return type is meaningless 
    add_definitions(-diag-disable 858) 
    
    # warning #1125: virtual function override intended 
    add_definitions(-diag-disable 1125) 

    if("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
      hpx_use_flag_if_available(ipo)
    elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
      hpx_use_flag_if_available(ipo)
    elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
      hpx_use_flag_if_available(ipo)
    else()
      hpx_error("build_type" "\"${CMAKE_BUILD_TYPE}\" is not a valid build type.")
    endif()
  endif()

  ##############################################################################
  # POSIX configuration tests 
  ##############################################################################
  # GCC's -march=native will automatically tune generated code for the host
  # environment. This is available on newish versions of GCC only (4.3ish). If
  # this flag is used, the generated binaries will be less portable. This is why
  # we define the HPX_COMPILER_AUTO_TUNED macro.
  hpx_check_for_compiler_auto_tune(HPX_COMPILER_AUTO_TUNE
     DEFINITIONS HPX_COMPILER_AUTO_TUNED) 
  
  if(HPX_COMPILER_AUTO_TUNE)
    hpx_append_flag("-march=native")
  endif() 

  # cmpxchg16b is an x86-64 extension present on most newer x86-64 machines.
  # It allows us to do double quadword (128bit) atomic compare and swap
  # operations, which is AWESOME. Note that early x86-64 processors do lack
  # this instruction.
  hpx_check_for_gnu_mcx16(HPX_GNU_MCX16
    DEFINITIONS HPX_HAVE_GNU_SYNC_16
                BOOST_ATOMIC_HAVE_GNU_SYNC_16) # for the gnu code

  if(HPX_GNU_MCX16)
    hpx_append_flag("-mcx16")
  endif() 

  # __attribute__ ((aligned(16))) should align a variable to a 16-byte, however,
  # GCC sets an upper limit on the maximum alignment (__BIGGEST_ALIGNMENT__)
  # and some versions don't warn if you ask for an alignment above said limit.
  # Instead, they'll just silently use the maximum, which can be problematical. 
  hpx_check_for_gnu_mcx16(HPX_GNU_ALIGNED_16
    DEFINITIONS HPX_HAVE_GNU_ALIGNED_16
                BOOST_ATOMIC_HAVE_GNU_ALIGNED_16) # for the gnu code

  # __uint128_t and __int128_t are a nifty, albeit undocumented, GNU extension
  # that's been supported in GCC (4.1ish and up) and clang-linux for awhile
  # (strangely, intel-linux doesn't support this). This is particularly useful
  # for use with cmpxchg16b
  hpx_check_for_gnu_128bit_integers(HPX_GNU_128BIT_INTEGERS
    DEFINITIONS HPX_HAVE_GNU_128BIT_INTEGERS
                BOOST_ATOMIC_HAVE_GNU_128BIT_INTEGERS) # for integral casts
 
  # we use movdqa for atomic 128bit loads and stores 
  hpx_cpuid("sse2" HPX_SSE2
    DEFINITIONS HPX_HAVE_SSE2
                BOOST_ATOMIC_HAVE_SSE2)
  
  if(HPX_SSE2)
    hpx_append_flag("-msse2")
  endif() 

  # rdtsc is an x86 instruction that reads the value of a CPU time stamp
  # counter. rdtscp is an extension to rdtsc. The difference is that rdtscp is
  # a serializing instruction.
  hpx_cpuid("rdtsc" HPX_RDTSC
    DEFINITIONS HPX_HAVE_RDTSC)
  hpx_cpuid("rdtscp" HPX_RDTSCP
    DEFINITIONS HPX_HAVE_RDTSCP)

  hpx_check_for_pthread_affinity_np(HPX_PTHREAD_AFFINITY_NP
    DEFINITIONS HPX_HAVE_PTHREAD_AFFINITY_NP)

  add_definitions(-D_GNU_SOURCE)
  hpx_append_flag(-pthread)

  set(hpx_LIBRARIES ${hpx_LIBRARIES} dl rt)

  set(HPX_MALLOC "TCMalloc" CACHE STRING
    "HPX malloc allocator (default: tcmalloc)" FORCE)

  find_package(HPX_TCMalloc)
  find_package(HPX_Jemalloc)

  if("${HPX_MALLOC}" MATCHES "tcmalloc|TCMalloc|TCMALLOC" AND NOT TCMALLOC_FOUND)
    hpx_warn("malloc" "tcmalloc allocator not found.")
  endif()

  if("${HPX_MALLOC}" MATCHES "jemalloc|Jemalloc|JEMALLOC" AND NOT JEMALLOC_FOUND)
    hpx_warn("malloc" "jemalloc allocator not found.")
  endif()
  
  set(hpx_MALLOC_LIBRARY "")

  if(NOT "${HPX_MALLOC}" MATCHES "system|System|SYSTEM")
    if(TCMALLOC_FOUND OR JEMALLOC_FOUND)
      if("${HPX_MALLOC}" MATCHES "tcmalloc|TCMalloc|TCMALLOC" OR NOT JEMALLOC_FOUND)
        hpx_info("malloc" "Using tcmalloc allocator.")
        set(hpx_MALLOC_LIBRARY ${TCMALLOC_LIBRARY})
        set(hpx_LIBRARIES ${hpx_LIBRARIES} ${TCMALLOC_LIBRARY})
        add_definitions(-DHPX_TCMALLOC)
      else()
        hpx_info("malloc" "Using jemalloc allocator.")
        set(hpx_MALLOC_LIBRARY ${JEMALLOC_LIBRARY})
        set(hpx_LIBRARIES ${hpx_LIBRARIES} ${JEMALLOC_LIBRARY})
        add_definitions(-DHPX_JEMALLOC)
      endif()
 
      hpx_use_flag_if_available(fno-builtin-cfree)
      hpx_use_flag_if_available(fno-builtin-pvalloc)
      hpx_use_flag_if_available(fno-builtin-malloc)
      hpx_use_flag_if_available(fno-builtin-free)
      hpx_use_flag_if_available(fno-builtin-calloc)
      hpx_use_flag_if_available(fno-builtin-realloc)
      hpx_use_flag_if_available(fno-builtin-valloc)
      hpx_use_flag_if_available(fno-builtin-memalign)
      hpx_use_flag_if_available(fno-builtin-posix_memalign)
    else()
      hpx_info("malloc" "Using system allocator.")
      hpx_warn("malloc" "HPX will perform poorly without tcmalloc.")
    endif()
  else()
    hpx_info("malloc" "Using system allocator.")
    hpx_warn("malloc" "HPX will perform poorly without tcmalloc.")
  endif()
endif()

################################################################################
# Mac OS X specific configuration 
################################################################################
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  hpx_warn("darwin_config" "HPX will perform poorly on Darwin.")
  add_definitions(-D_XOPEN_SOURCE=1) # for some reason Darwin whines without this
endif()

################################################################################
# target specification 
################################################################################
# hpx_LIBRARIES lists all libraries a HPX module needs to be linked with by 
# default. There may be more for individual modules, but this is the mandatory
# set.
set(hpx_CORE hpx hpx_serialization)
set(hpx_LIBRARIES ${hpx_LIBRARIES} hpx hpx_serialization)

# Recurse into some subdirectories. This does not actually cause another cmake 
# executable to run. The same process will walk through the project's entire 
# directory structure.
add_subdirectory(src)

if(MSVC)
  option(HPX_EXAMPLES "Build HPX examples (default: OFF)" OFF)
else()
  option(HPX_EXAMPLES "Build HPX examples (default: ON)" ON)
endif()

if(HPX_EXAMPLES)
  # executables that depend on libraries above
  add_hpx_pseudo_target(examples)

  include_directories(examples)
  add_subdirectory(examples)
endif()

add_subdirectory(runtime)

if(NOT MSVC)
  add_hpx_pseudo_target(tools) 
  add_subdirectory(tools)
endif()

################################################################################
# test configuration 
################################################################################
if(NOT MSVC)
  set(DART_TESTING_TIMEOUT "60" CACHE STRING
    "CTest timeout, in seconds (Default: 60).")

  enable_testing()
  include(CTest)
  add_hpx_pseudo_target(tests) 
  add_subdirectory(tests)
endif()

################################################################################
# installation instructions
################################################################################
install( # install all hpx header files
  DIRECTORY hpx/ 
  DESTINATION include/hpx 
  COMPONENT core
  FILES_MATCHING PATTERN "*.h*"
  PATTERN ".svn" EXCLUDE
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE)

install( # install all hpx cmake utility files
  DIRECTORY cmake/ 
  DESTINATION share/hpx/cmake 
  COMPONENT core
  PATTERN ".svn" EXCLUDE
  PATTERN ".git" EXCLUDE)

install( # install hpx python scripts 
  DIRECTORY python/scripts/ 
  DESTINATION bin
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                   GROUP_READ GROUP_EXECUTE
                   WORLD_READ WORLD_EXECUTE
  COMPONENT core
  FILES_MATCHING PATTERN "*.py"
  PATTERN ".svn" EXCLUDE
  PATTERN ".git" EXCLUDE)

if(UNIX)
  file(GLOB scripts "${CMAKE_SOURCE_DIR}/python/scripts/*.py") 
  foreach(script ${scripts})
    get_filename_component(script_name ${script} NAME)
    get_filename_component(script_name_we ${script} NAME_WE)
    install(CODE
      "set(bindir \"${CMAKE_INSTALL_PREFIX}/bin/\")
       execute_process(
         COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink
                 \"${script_name}\" \"${script_name_we}\"
         WORKING_DIRECTORY \"\${bindir}\")")
  endforeach()
endif()

install( # install hpx python module (TODO: this is a temporary hack) 
  DIRECTORY python/hpx 
  DESTINATION share/hpx/python 
  COMPONENT core
  FILES_MATCHING PATTERN "*.py"
  PATTERN ".svn" EXCLUDE
  PATTERN ".git" EXCLUDE)

install( # install external dependencies
  DIRECTORY external/cache/boost
            external/coroutine/boost
            external/atomic/boost
            external/endian/boost
            external/lockfree/boost
            external/logging/boost
            external/plugin/boost
  DESTINATION include
  COMPONENT core
  FILES_MATCHING PATTERN "*.h*"
  PATTERN ".svn" EXCLUDE
  PATTERN ".git" EXCLUDE)

if(HPX_INTERNAL_CHRONO AND ${BOOST_MINOR_VERSION} LESS 47)
  install( # install Boost.Chrono headers if we're using our internal version
    DIRECTORY external/chrono/boost
    DESTINATION include
    COMPONENT core
    FILES_MATCHING PATTERN "*.h*"
    PATTERN ".svn" EXCLUDE
    PATTERN ".git" EXCLUDE)
endif()

