
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig)
# look for cray pmi...
pkg_check_modules(PC_PMI_CRAY QUIET cray-pmi)
# look for the rest if we couldn't find the cray package
if (NOT PC_PMI_CRAY_FOUND)
  pkg_check_modules(PC_PMI QUIET pmi)
endif()


find_path(PMI_INCLUDE_DIR pmi2.h
  HINTS
    ${PMI_ROOT} ENV PMI_ROOT
    ${PMI_DIR} ENV PMI_DIR
    ${PC_PMI_CRAY_INCLUDEDIR}
    ${PC_PMI_CRAY_INCLUDE_DIRS}
    ${PC_PMI_INCLUDEDIR}
    ${PC_PMI_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(PMI_LIBRARY NAMES pmi
  HINTS
    ${PMI_ROOT} ENV PMI_ROOT
    ${PC_PMI_CRAY_LIBDIR}
    ${PC_PMI_CRAY_LIBRARY_DIRS}
    ${PC_PMI_LIBDIR}
    ${PC_PMI_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(PMI_LIBRARIES ${PMI_LIBRARY} CACHE INTERNAL "")
set(PMI_INCLUDE_DIRS ${PMI_INCLUDE_DIR} CACHE INTERNAL "")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PMI DEFAULT_MSG
  PMI_LIBRARY PMI_INCLUDE_DIR)

#foreach(v PMI_ROOT)
#  get_property(_type CACHE ${v} PROPERTY TYPE)
#  if(_type)
#    set_property(CACHE ${v} PROPERTY ADVANCED 1)
#    if("x${_type}" STREQUAL "xUNINITIALIZED")
#      set_property(CACHE ${v} PROPERTY TYPE PATH)
#    endif()
#  endif()
#endforeach()

mark_as_advanced(PMI_ROOT PMI_LIBRARY PMI_INCLUDE_DIR)
