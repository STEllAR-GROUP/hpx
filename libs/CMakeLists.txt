# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This file is auto generated. Please do not edit manually.

include(HPX_Message)

set(HPX_LIBS
  cache
  config
  preprocessor
  CACHE INTERNAL "" FORCE
)

hpx_info("Configuring modules:")

set(MODULE_FORCE_LINKING_INCLUDES)
set(MODULE_FORCE_LINKING_CALLS)
foreach(lib ${HPX_LIBS})
  add_subdirectory(${lib})

  set(MODULE_FORCE_LINKING_INCLUDES
    "${MODULE_FORCE_LINKING_INCLUDES}#include <hpx/${lib}/force_linking.hpp>\n")

  set(MODULE_FORCE_LINKING_CALLS
    "${MODULE_FORCE_LINKING_CALLS}\n        ${lib}::force_linking();")

endforeach()

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/templates/modules.cpp.in"
    "${CMAKE_BINARY_DIR}/libs/modules.cpp"
    @ONLY)
