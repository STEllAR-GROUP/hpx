# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(components)

if(UNIX AND HPX_WARNINGS)
  hpx_use_flag_if_available(-Wextra)

  # We never want trampolines
  hpx_use_flag_if_available(-Wtrampolines)

  # These are usually benign and can't be suppressed because of
  # interface requirements
  hpx_use_flag_if_available(-Wno-unused-parameter)

  # There are ignored qualifiers in Boost, so we have to ignore them
  hpx_use_flag_if_available(-Wno-ignored-qualifiers)

  # Be extra strict about format checks
  # Boost.Logging is built on fprintf, sadly
  hpx_use_flag_if_available(-Wformat=2)
  hpx_use_flag_if_available(-Wno-format-nonliteral)

  # Self initialization is dangerous
  hpx_use_flag_if_available(-Winit-self)

  # For portability
  hpx_use_flag_if_available(-Wdouble-promotion)

  # Warn about casting that violates qualifiers or alignment
  hpx_use_flag_if_available(-Wcast-qual)
  hpx_use_flag_if_available(-Wcast-align)

  hpx_use_flag_if_available(-Werror=trampolines)
  hpx_use_flag_if_available(-Werror=parentheses)
  hpx_use_flag_if_available(-Werror=reorder)
  hpx_use_flag_if_available(-Werror=return-type)
  hpx_use_flag_if_available(-Werror=sequence-point)

  if(040600 LESS ${GCC_VERSION} OR 040600 EQUAL ${GCC_VERSION})
    # Uninitialized variables are bad, earlier compilers issue spurious warnings
    hpx_use_flag_if_available(-Werror=uninitialized)
  endif()

  hpx_use_flag_if_available(-Werror=missing-field-initializers)
  hpx_use_flag_if_available(-Werror=format)
  hpx_use_flag_if_available(-Werror=missing-braces)
  hpx_use_flag_if_available(-Werror=sign-compare)
endif()

foreach(lib ${hpx_LIBRARIES})
  set(${lib}_SOURCES "" CACHE INTERNAL "Sources for lib${lib}." FORCE)

  if(MSVC)
    set(${lib}_HEADERS "" CACHE INTERNAL "Headers for lib${lib}." FORCE)
  endif()
endforeach()

################################################################################
# gather sources

# libhpx_serialization sources
add_hpx_library_sources(hpx_serialization
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/util/portable_binary_*archive.cpp")

# libhpx sources
add_hpx_library_sources(hpx
  GLOB GLOBS "${hpx_SOURCE_DIR}/src/*.cpp"
  EXCLUDE "(.*(hpx_main|hpx_init|hpx_user).*[.]cpp)|main.cpp")
add_hpx_library_sources(hpx
  GLOB GLOBS "${hpx_SOURCE_DIR}/src/pre_main.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/runtime/*.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/performance_counters/*.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/util/*.cpp"
  EXCLUDE ".*portable_binary_.*archive[.]cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/lcos/*.cpp"
  APPEND)

# libhpx_init sources
add_hpx_library_sources(hpx_init
  GLOB GLOBS "${hpx_SOURCE_DIR}/src/hpx_*.cpp")
add_hpx_library_sources(hpx_init
  GLOB GLOBS "${hpx_SOURCE_DIR}/src/main.cpp" APPEND)

set(hpx_external_SOURCES "" CACHE INTERNAL "Sources for libhpx_external." FORCE)

# Boost Chrono sources (as long as this library is not in Boost)
if(HPX_INTERNAL_CHRONO)
  add_hpx_library_sources(hpx_external
    GLOB GLOBS "${hpx_SOURCE_DIR}/external/chrono/libs/chrono/src/chrono.cpp"
    APPEND)
endif()

# Add Boost.Backtrace sources if stack traces are enabled
if(HPX_STACKTRACES)
  add_hpx_library_sources(hpx_external
    GLOB GLOBS "${hpx_SOURCE_DIR}/external/backtrace/libs/backtrace/src/backtrace.cpp"
    APPEND)
endif()

if(NOT MSVC)
  add_hpx_library_sources(hpx_external
    GLOB GLOBS "${hpx_SOURCE_DIR}/external/coroutine/libs/coroutine/src/swapcontext.cpp"
    APPEND)
else()
  if(HPX_EMULATE_SWAP_CONTEXT)
    set(switch_to_fiber_source "${hpx_SOURCE_DIR}/external/coroutine/libs/coroutine/src/switch_to_fiber.asm")
    add_hpx_library_sources(hpx_external_objects
      GLOB GLOBS "${switch_to_fiber_source}")
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/switch_to_fiber.obj
      COMMAND ${CMAKE_ASM_MASM_COMPILER} /Fo ${CMAKE_CURRENT_BINARY_DIR}/switch_to_fiber.obj /nologo /c ${switch_to_fiber_source}
      DEPENDS ${switch_to_fiber_source})
    set(hpx_external_OBJECTS ${CMAKE_CURRENT_BINARY_DIR}/switch_to_fiber.obj
      CACHE INTERNAL "External object files for HPX." FORCE)
    add_definitions(-DHPX_EMULATE_SWAP_CONTEXT)
  endif()
endif()

if(OPENCL_FOUND)
  add_hpx_library_sources(hpx_external
    GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/external/oclm/src/*.cpp"
    APPEND)
endif()

################################################################################
# gather headers

# libhpx_serialization headers
add_hpx_library_headers(hpx_serialization
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/util/portable_binary_*archive.hpp")

# libhpx headers
add_hpx_library_headers(hpx
  GLOB GLOBS "${hpx_SOURCE_DIR}/hpx/*.hpp"
  EXCLUDE ".*(hpx_main|hpx_init).hpp")
add_hpx_library_headers(hpx
  GLOB GLOBS "${hpx_SOURCE_DIR}/hpx/preprocessed/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/runtime/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/config/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/include/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/lcos/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/performance_counters/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/traits/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/util/*.h*"
  EXCLUDE ".*portable_binary_.*archive[.]hpp"
  APPEND)

# libhpx_init headers
add_hpx_library_headers(hpx_init
  GLOB GLOBS "${hpx_SOURCE_DIR}/hpx/hpx_init.hpp")
add_hpx_library_headers(hpx_init
  GLOB GLOBS "${hpx_SOURCE_DIR}/hpx/hpx_main.hpp"
  APPEND)

################################################################################
# add headers to preprocessing
hpx_partial_preprocess_header(
    "hpx/util/detail/function_template.hpp"
    LIMIT HPX_FUNCTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/util/detail/get_table.hpp"
    LIMIT HPX_FUNCTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/util/detail/vtable.hpp"
    LIMIT HPX_FUNCTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/util/detail/vtable_ptr.hpp"
    LIMIT HPX_FUNCTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/util/detail/vtable_ptr_base.hpp"
    LIMIT HPX_FUNCTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/util/bind.hpp"
    LIMIT HPX_FUNCTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/util/bind_action.hpp"
    LIMIT HPX_FUNCTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/util/tuple.hpp"
    LIMIT HPX_TUPLE_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/util/locking_helpers.hpp"
    LIMIT HPX_LOCK_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/apply.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/async.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/lcos/async_fwd.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/lcos/wait_all.hpp"
    LIMIT HPX_WAIT_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/lcos/wait_n.hpp"
    LIMIT HPX_WAIT_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/lcos/wait_any.hpp"
    LIMIT HPX_WAIT_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/lcos/future_wait.hpp"
    LIMIT HPX_WAIT_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/lcos/packaged_action_constructors.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/lcos/packaged_action_constructors_direct.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/lcos/deferred_packaged_task_constructors.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/lcos/deferred_packaged_task_constructors_direct.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/runtime/actions/component_action_enum.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/runtime/actions/component_action_implementations.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/runtime/actions/construct_continuation_function_objects.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/runtime/actions/construct_continuation_functions.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/runtime/actions/define_function_operators.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/runtime/actions/plain_action_enum.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/runtime/actions/plain_action_implementations.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/runtime/actions/transfer_action_constructors.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/runtime/applier/apply_implementations.hpp"
    LIMIT HPX_ACTION_ARGUMENT_LIMIT
)
hpx_partial_preprocess_header(
    "hpx/runtime/components/memory_block.hpp"
    LIMIT HPX_WAIT_ARGUMENT_LIMIT
)

################################################################################

################################################################################
# make source groups
add_hpx_source_group(
  NAME hpx_serialization
  CLASS "Source Files"
  ROOT ${hpx_SOURCE_DIR}/src/util
  TARGETS ${hpx_serialization_SOURCES})

add_hpx_source_group(
  NAME hpx_serialization
  CLASS "Header Files"
  ROOT ${hpx_SOURCE_DIR}/hpx/util
  TARGETS ${hpx_serialization_HEADERS})

add_hpx_source_group(
  NAME hpx CLASS "Source Files"
  ROOT ${hpx_SOURCE_DIR}/src
  TARGETS ${hpx_SOURCES})

add_hpx_source_group(
  NAME hpx CLASS "External Source Files"
  ROOT ${hpx_SOURCE_DIR}/external
  TARGETS ${hpx_external_SOURCES})

add_hpx_source_group(
  NAME hpx
  CLASS "Header Files"
  ROOT ${hpx_SOURCE_DIR}/hpx
  TARGETS ${hpx_HEADERS})

add_hpx_source_group(
  NAME hpx_init
  CLASS "Source Files"
  ROOT ${hpx_SOURCE_DIR}/src
  TARGETS ${hpx_init_SOURCES})

add_hpx_source_group(
  NAME hpx_init
  CLASS "Header Files"
  ROOT ${hpx_SOURCE_DIR}/hpx
  TARGETS ${hpx_init_HEADERS})

################################################################################
# libhpx_serialization
add_library(hpx_serialization SHARED
  ${hpx_serialization_SOURCES} ${hpx_serialization_HEADERS})

if(NOT MSVC)
  set_target_properties(hpx_serialization
                        PROPERTIES SKIP_BUILD_RPATH TRUE
                                   BUILD_WITH_INSTALL_RPATH TRUE
                                   INSTALL_RPATH_USE_LINK_PATH TRUE
                                   INSTALL_RPATH ${HPX_RPATH})
endif()

set_target_properties(hpx_serialization PROPERTIES
  VERSION ${HPX_VERSION}
  SOVERSION ${HPX_SOVERSION}
  CLEAN_DIRECT_OUTPUT 1
  OUTPUT_NAME hpx_serialization
  FOLDER "Core")

if(MSVC)
  set_target_properties(hpx_serialization PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${HPX_LIBRARY_OUTPUT_DIRECTORY_RELEASE}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${HPX_LIBRARY_OUTPUT_DIRECTORY_DEBUG}
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${HPX_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${HPX_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${HPX_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${HPX_ARCHIVE_OUTPUT_DIRECTORY_DEBUG}
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${HPX_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL}
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${HPX_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${HPX_LIBRARY_OUTPUT_DIRECTORY_RELEASE}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${HPX_LIBRARY_OUTPUT_DIRECTORY_DEBUG}
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${HPX_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL}
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${HPX_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO})
else()
  set_target_properties(hpx_serialization PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${HPX_RUNTIME_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${HPX_LIBRARY_OUTPUT_DIRECTORY})
endif()

set(hpx_serialization_lib_dependencies
  ${hpx_MALLOC_LIBRARY})

if(NOT MSVC)
  set(hpx_serialization_lib_dependencies
      ${hpx_serialization_lib_dependencies}
      ${BOOST_SYSTEM_LIBRARY}
      ${BOOST_SERIALIZATION_LIBRARY})
endif()

target_link_libraries(hpx_serialization
  ${hpx_serialization_lib_dependencies})

hpx_get_target_file(name hpx_serialization)

set_property(TARGET hpx_serialization APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_LIBRARY=\"${name}\""
  "HPX_EXPORTS"
  "BOOST_COROUTINE_EXPORTS")

if(NOT HPX_NO_INSTALL)
  hpx_library_install(hpx_serialization)
endif()

################################################################################
# libhpx
add_library(hpx SHARED
  ${hpx_SOURCES} ${hpx_external_SOURCES} ${hpx_external_OBJECTS} ${hpx_HEADERS})

set_target_properties(hpx PROPERTIES
  VERSION ${HPX_VERSION}
  SOVERSION ${HPX_SOVERSION}
  CLEAN_DIRECT_OUTPUT 1
  OUTPUT_NAME hpx
  FOLDER "Core")

if(MSVC)
  set_target_properties(hpx PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${HPX_LIBRARY_OUTPUT_DIRECTORY_RELEASE}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${HPX_LIBRARY_OUTPUT_DIRECTORY_DEBUG}
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${HPX_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${HPX_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${HPX_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${HPX_ARCHIVE_OUTPUT_DIRECTORY_DEBUG}
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${HPX_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL}
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${HPX_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${HPX_LIBRARY_OUTPUT_DIRECTORY_RELEASE}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${HPX_LIBRARY_OUTPUT_DIRECTORY_DEBUG}
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${HPX_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL}
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${HPX_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO})
else()
  set_target_properties(hpx PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${HPX_RUNTIME_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${HPX_LIBRARY_OUTPUT_DIRECTORY})
endif()

if(NOT MSVC)
  set_target_properties(hpx
                        PROPERTIES SKIP_BUILD_RPATH TRUE
                                   BUILD_WITH_INSTALL_RPATH TRUE
                                   INSTALL_RPATH_USE_LINK_PATH TRUE
                                   INSTALL_RPATH ${HPX_RPATH})
endif()

if(MSVC AND HPX_EMULATE_SWAP_CONTEXT)
  set_target_properties(hpx PROPERTIES LINK_FLAGS "/EXPORT:switch_to_fiber")
endif()

target_link_libraries(hpx
  hpx_serialization
  ${hpx_MALLOC_LIBRARY}
  ${hpx_RUNTIME_LIBRARIES}
  ${BOOST_FOUND_LIBRARIES})

hpx_get_target_file(name hpx)

set_property(TARGET hpx APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_LIBRARY=\"${name}\""
  "HPX_EXPORTS"
  "HPX_ACTION_ARGUMENT_LIMIT=4"
  "HPX_FUNCTION_ARGUMENT_LIMIT=7"
  "BOOST_ENABLE_ASSERT_HANDLER"
  "BOOST_COROUTINE_EXPORTS") # Do we still need this?

if(NOT HPX_NO_INSTALL)
  hpx_library_install(hpx)
endif()

################################################################################
# libhpx_init
add_library(hpx_init STATIC
  ${hpx_init_SOURCES} ${hpx_init_HEADERS})

set_property(TARGET hpx_init APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_APPLICATION_EXPORTS"
  "HPX_ACTION_ARGUMENT_LIMIT=4"
  "HPX_FUNCTION_ARGUMENT_LIMIT=7"
  "BOOST_ENABLE_ASSERT_HANDLER")

if(MSVC)
  set_target_properties(hpx_init PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${HPX_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${HPX_ARCHIVE_OUTPUT_DIRECTORY_DEBUG}
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${HPX_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL}
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${HPX_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO})
else()
  set_target_properties(hpx_init PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${HPX_ARCHIVE_OUTPUT_DIRECTORY})
endif()

set_property(TARGET hpx_init PROPERTY FOLDER "Core")

if(NOT HPX_NO_INSTALL)
  hpx_archive_install(hpx_init)
endif()

