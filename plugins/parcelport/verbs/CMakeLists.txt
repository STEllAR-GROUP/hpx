# Copyright (c) 2014-2015 John Biddiscombe
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

  ## ibverbs parcelport settings
  hpx_option(HPX_PARCELPORT_VERBS_IFNAME STRING "The interface name of the ibverbs capable network adapter (default: ib0)" "ib0" CATEGORY "Parcelport" ADVANCED)
  hpx_option(HPX_PARCELPORT_VERBS_DEVICE STRING "The device name of the ibverbs capable network adapter (default: roq0)" "roq" CATEGORY "Parcelport" ADVANCED)
  hpx_option(HPX_PARCELPORT_VERBS_INTERFACE STRING "The interface name of the ibverbs capable network adapter (default: roq0)" "roq0" CATEGORY "Parcelport" ADVANCED)
  hpx_option(HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE STRING "Number of bytes a chunk in the memory pool can hold (default: 4K)" "4096" CATEGORY "Parcelport" ADVANCED)
  hpx_option(HPX_PARCELPORT_VERBS_MAX_MEMORY_CHUNKS STRING "Maximum number of chunks that can be allocated (default: 16)" "64" CATEGORY "Parcelport" ADVANCED)

  hpx_add_config_define(HPX_PARCELPORT_VERBS)
  hpx_add_config_define(HPX_PARCELPORT_VERBS_IFNAME "\"${HPX_PARCELPORT_VERBS_IFNAME}\"")
  hpx_add_config_define(HPX_PARCELPORT_VERBS_DEVICE "\"${HPX_PARCELPORT_VERBS_DEVICE}\"")
  hpx_add_config_define(HPX_PARCELPORT_VERBS_INTERFACE "\"${HPX_PARCELPORT_VERBS_INTERFACE}\"")
  hpx_add_config_define(HPX_PARCELPORT_VERBS_MESSAGE_PAYLOAD "${HPX_PARCELPORT_VERBS_MESSAGE_PAYLOAD}")
  hpx_add_config_define(HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE "${HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE}")
  hpx_add_config_define(HPX_PARCELPORT_VERBS_MAX_MEMORY_CHUNKS "${HPX_PARCELPORT_VERBS_MAX_MEMORY_CHUNKS}")

  message("Adding subproject")
  set(rdmahelper_EXPORTED_TARGETS HPXTargets)
  
  # build rdma helperlibrary with HPX support
  add_definitions(-DRDMAHELPER_HPX_COMPATIBILITY)
  
  add_subproject(rdmahelper rdmahelper)

  find_package(rdmahelper REQUIRED)
  
  set(_parcelport_verbs_SOURCES 
    "${hpx_SOURCE_DIR}/plugins/parcelport/verbs/parcelport_verbs.cpp" 
    CACHE INTERNAL "global variable" FORCE)
  set(_parcelport_verbs_INCLUDE_DIRECTORIES 
    "${rdmahelper_INCLUDE_DIRS}" 
    "${CMAKE_CURRENT_SOURCE_DIR}"
    CACHE INTERNAL "global variable" FORCE)
  set(_parcelport_verbs_LINKMODE 
    "STATIC" 
    CACHE INTERNAL "global variable" FORCE)
  set(_parcelport_verbs_LIBRARIES 
    "rdmahelper" 
    CACHE INTERNAL "global variable" FORCE)

  macro(add_parcelport_verbs_module)
    hpx_debug("add_parcelport_verbs_module")

    add_parcelport(verbs
      STATIC
      SOURCES
        "${hpx_SOURCE_DIR}/plugins/parcelport/verbs/parcelport_verbs.cpp"
      INCLUDE_DIRECTORIES
        "${_parcelport_verbs_INCLUDE_DIRECTORIES}"
      HEADERS
      DEPENDENCIES
      FOLDER 
        "Core/Plugins/Parcelport/verbs")
  endmacro()
else()
  macro(add_parcelport_verbs_module)
  endmacro()
endif()
