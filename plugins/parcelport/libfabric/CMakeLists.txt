# Copyright (c) 2014-2016 John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_SetupLibfabric)

macro(add_parcelport_libfabric_module)
  if (HPX_WITH_PARCELPORT_LIBFABRIC)
    hpx_debug("add_parcelport_libfabric_module")

    add_parcelport(libfabric
      STATIC
      SOURCES
          ${_parcelport_CXX_srcs}
          ${_libfabric_C_srcs}
      # do we need to declare any headers?
      #HEADERS
      DEPENDENCIES
        hpx::libfabric
      FOLDER
        "Core/Plugins/Parcelport/libfabric")

    # FIXME : Could this be private
    target_include_directories(libfabric PUBLIC ${_libfabric_include_dirs})

    if(HPX_PARCELPORT_LIBFABRIC_WITH_HPXC)
      set_source_files_properties(
        ${_libfabric_C_srcs}
        PROPERTIES
          COMPILE_FLAGS
              "-include \"hpxc/hpxc/pthread_compatibility.h\" \
              -include \"libfabric/config.h\"                 \
              -Wsign-compare -Wunused-parameter"
          LANGUAGE C)

      set_source_files_properties(
        ${_parcelport_CXX_srcs}
        PROPERTIES
          COMPILE_FLAGS "-include \"libfabric/config.h\" -Wno-missing-field-initializers"
          LANGUAGE CXX)
    else()
      set_source_files_properties(
        ${_parcelport_CXX_srcs}
        PROPERTIES
          COMPILE_FLAGS "-Wno-missing-field-initializers"
          LANGUAGE CXX)
    endif()
  endif()
endmacro()

