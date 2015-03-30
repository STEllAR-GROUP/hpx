# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2014-2015 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

################################################################################
# Decide whether to use the VERBS based parcelport
################################################################################
if(HPX_PARCELPORT_VERBS)
  hpx_add_config_define(HPX_PARCELPORT_VERBS)

  include(GitExternal)
  include(SubProject)
  
  #--------------------------------------------------
  # checkout sub-project required by this plugin
  #--------------------------------------------------
  Git_External(
    ${CMAKE_CURRENT_SOURCE_DIR}/rdmahelper
    git@github.com:biddisco/rdmahelper.git
    master
    VERBOSE
    NO_UPDATE
  )

  message("Adding subproject")
  add_subproject(rdmahelper rdmahelper)

  # this parcelport is defined entirely locally, 
  # so headers are not part of the public API
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  
  macro(add_parcelport_verbs_module)
    hpx_debug("add_parcelport_verbs_module")

    add_parcelport(verbs
      STATIC
      SOURCES
        "${hpx_SOURCE_DIR}/plugins/parcelport/verbs/parcelport_verbs.cpp"
      HEADERS
      DEPENDENCIES
      FOLDER "Core/Plugins/Parcelport/verbs")
  endmacro()
else()
  macro(add_parcelport_verbs_module)
  endmacro()
endif()
