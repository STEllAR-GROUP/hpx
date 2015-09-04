# Copyright (c) 2014-2015 John Biddiscombe
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

################################################################################
# Decide whether to use the VERBS based parcelport
################################################################################
if(HPX_WITH_PARCELPORT_VERBS)

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

  if (NOT Boost_USE_STATIC_LIBS)
    hpx_add_config_define(BOOST_LOG_DYN_LINK)
  endif()

## ibverbs parcelport settings
#hpx_option(HPX_PARCELPORT_VERBS_IFNAME STRING
#  "The interface name of the ibverbs capable network adapter (default: ib0)"
#  "ib0" CATEGORY "Parcelport" ADVANCED)

#hpx_option(HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE STRING
#  "Number of bytes a chunk in the memory pool can hold (default: 64MB)"
#  "67108864" CATEGORY "Parcelport" ADVANCED)

hpx_option(HPX_PARCELPORT_VERBS_MESSAGE_PAYLOAD STRING
  "Size of the message payload not sent with RDMA (default: 512 byte)"
  "512" CATEGORY "Parcelport" ADVANCED)

hpx_option(HPX_PARCELPORT_VERBS_DEVICE STRING 
  "The device name of the ibverbs capable network adapter (default: roq0)" 
  "roq" CATEGORY "Parcelport" ADVANCED)

hpx_option(HPX_PARCELPORT_VERBS_INTERFACE STRING 
  "The interface name of the ibverbs capable network adapter (default: roq0)" 
  "roq0" CATEGORY "Parcelport" ADVANCED)

hpx_option(HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE STRING 
  "Number of bytes a chunk in the memory pool can hold (default: 4K)" 
  "4096" CATEGORY "Parcelport" ADVANCED)
  
hpx_option(HPX_PARCELPORT_VERBS_DEFAULT_MEMORY_CHUNKS STRING 
  "The number of chunks that can are allocated initially (default: 64)" 
  "64" CATEGORY "Parcelport" ADVANCED)

hpx_option(HPX_PARCELPORT_VERBS_MAX_MEMORY_CHUNKS STRING 
  "Maximum number of chunks that can be allocated (default: 128)" 
  "128" CATEGORY "Parcelport" ADVANCED)

hpx_option(HPX_PARCELPORT_VERBS_MAX_PREPOSTS STRING 
  "The number of pre-posted receive buffers (default: 32)" 
  "32" CATEGORY "Parcelport" ADVANCED)

  hpx_add_config_define(HPX_HAVE_PARCELPORT_VERBS)
#  hpx_add_config_define(HPX_PARCELPORT_VERBS_IFNAME "\"${HPX_PARCELPORT_VERBS_IFNAME}\"")
  hpx_add_config_define(HPX_PARCELPORT_VERBS_DEVICE "\"${HPX_PARCELPORT_VERBS_DEVICE}\"")
  hpx_add_config_define(HPX_PARCELPORT_VERBS_INTERFACE "\"${HPX_PARCELPORT_VERBS_INTERFACE}\"")
  hpx_add_config_define(HPX_PARCELPORT_VERBS_MESSAGE_PAYLOAD "${HPX_PARCELPORT_VERBS_MESSAGE_PAYLOAD}")
  hpx_add_config_define(HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE "${HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE}")
  hpx_add_config_define(HPX_PARCELPORT_VERBS_DEFAULT_MEMORY_CHUNKS "${HPX_PARCELPORT_VERBS_DEFAULT_MEMORY_CHUNKS}")
  hpx_add_config_define(HPX_PARCELPORT_VERBS_MAX_MEMORY_CHUNKS "${HPX_PARCELPORT_VERBS_MAX_MEMORY_CHUNKS}")
  hpx_add_config_define(HPX_PARCELPORT_VERBS_MAX_PREPOSTS "${HPX_PARCELPORT_VERBS_MAX_PREPOSTS}")


  if(HPX_PARCELPORT_VERBS_INTERFACE MATCHES "roq")
    message(WARNING "The roq interface on BGQ IO nodes does not support immediate data, disabling it")
    hpx_add_config_define(HPX_PARCELPORT_VERBS_IMM_UNSUPPORTED 1)
  endif()
  
  message("Adding subproject")
  set(rdmahelper_EXPORTED_TARGETS HPXTargets)
 
  if (RDMAHELPER_WITH_LOGGING)
    set_source_files_properties(
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/parcelport_verbs.cpp"
      PROPERTIES COMPILE_DEFINITIONS "RDMAHELPER_HAVE_LOGGING"
    )
  endif()

  # build rdma helperlibrary with HPX support
  add_definitions(-DRDMAHELPER_HAVE_HPX)
  
  add_subproject(rdmahelper rdmahelper)

  find_package(rdmahelper REQUIRED)
  
  set(_parcelport_verbs_SOURCES 
    "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/parcelport_verbs.cpp" 
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
        "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/parcelport_verbs.cpp"
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
