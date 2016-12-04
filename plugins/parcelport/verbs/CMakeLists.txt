# Copyright (c) 2014-2016 John Biddiscombe
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#------------------------------------------------------------------------------
# If not enabled, define the empty registration macro and exit
#------------------------------------------------------------------------------
if (NOT HPX_WITH_PARCELPORT_VERBS)
  macro(add_parcelport_verbs_module)
  endmacro()
  return()
endif()

#------------------------------------------------------------------------------
# Add #define to global defines.hpp
#------------------------------------------------------------------------------
hpx_add_config_define(HPX_HAVE_PARCELPORT_VERBS)

#------------------------------------------------------------------------------
# OFED verbs stack
#------------------------------------------------------------------------------
find_package(IB_VERBS REQUIRED)

#------------------------------------------------------------------------------
# RDMA_CM
#------------------------------------------------------------------------------
find_package(RDMA_CM REQUIRED)

#------------------------------------------------------------------------------
# Logging
#------------------------------------------------------------------------------
hpx_option(HPX_PARCELPORT_VERBS_WITH_LOGGING BOOL
  "Enable logging in the Verbs ParcelPort "
  "(default: OFF - Warning - severely impacts usability when enabled)"
  OFF CATEGORY "Parcelport" ADVANCED)

if (HPX_PARCELPORT_VERBS_WITH_LOGGING)
  hpx_add_config_define_namespace(
      DEFINE    HPX_PARCELPORT_VERBS_HAVE_LOGGING
      NAMESPACE parcelport_verbs)
endif()

#------------------------------------------------------------------------------
# Development mode (extra logging and customizable tweaks)
#------------------------------------------------------------------------------
hpx_option(HPX_PARCELPORT_VERBS_WITH_DEV_MODE BOOL
  "Enables some extra logging and debug features in the verbs parcelport "
  "(default: OFF - Warning - severely impacts usability when enabled)"
  OFF CATEGORY "Parcelport" ADVANCED)

if (HPX_PARCELPORT_VERBS_WITH_DEV_MODE)
  hpx_add_config_define_namespace(
      DEFINE    HPX_PARCELPORT_VERBS_HAVE_DEV_MODE
      NAMESPACE parcelport_verbs)
endif()

#------------------------------------------------------------------------------
# make sure boost log is linked correctly
#------------------------------------------------------------------------------
if(HPX_PARCELPORT_VERBS_WITH_LOGGING OR HPX_PARCELPORT_VERBS_WITH_DEV_MODE)
  if (NOT Boost_USE_STATIC_LIBS)
    hpx_add_config_define_namespace(
        DEFINE    BOOST_LOG_DYN_LINK
        NAMESPACE parcelport_verbs)
  endif()
endif()

#------------------------------------------------------------------------------
# IB device selection
#------------------------------------------------------------------------------
hpx_option(HPX_PARCELPORT_VERBS_DEVICE STRING
  "The device name of the ibverbs capable network adapter (default: mlx5_0)"
  "mlx5_0" CATEGORY "Parcelport" ADVANCED)

hpx_option(HPX_PARCELPORT_VERBS_INTERFACE STRING
  "The interface name of the ibverbs capable network adapter (default: ib0)"
  "ib0" CATEGORY "Parcelport" ADVANCED)

hpx_add_config_define_namespace(
    DEFINE HPX_PARCELPORT_VERBS_DEVICE
    VALUE  "\"${HPX_PARCELPORT_VERBS_DEVICE}\""
    NAMESPACE parcelport_verbs)

hpx_add_config_define_namespace(
    DEFINE HPX_PARCELPORT_VERBS_INTERFACE
    VALUE  "\"${HPX_PARCELPORT_VERBS_INTERFACE}\""
    NAMESPACE parcelport_verbs)

#------------------------------------------------------------------------------
# Bootstrap options
#------------------------------------------------------------------------------
hpx_option(HPX_PARCELPORT_VERBS_WITH_BOOTSTRAPPING BOOL
  "Configure the parcelport to enable bootstrap capabilities (default: ON)"
  ON CATEGORY "Parcelport" ADVANCED)

if (HPX_PARCELPORT_VERBS_WITH_BOOTSTRAPPING)
  hpx_add_config_define_namespace(
      DEFINE    HPX_PARCELPORT_VERBS_HAVE_BOOTSTRAPPING
      VALUE     std::true_type
      NAMESPACE parcelport_verbs)
else()
  hpx_add_config_define_namespace(
      DEFINE    HPX_PARCELPORT_VERBS_HAVE_BOOTSTRAPPING
      VALUE     std::false_type
      NAMESPACE parcelport_verbs)
endif()

#------------------------------------------------------------------------------
# Performance counters
#------------------------------------------------------------------------------
hpx_option(HPX_PARCELPORT_VERBS_WITH_PERFORMANCE_COUNTERS BOOL
  "Enable verbs parcelport performance counters (default: OFF)"
  OFF CATEGORY "Parcelport" ADVANCED)

if (HPX_PARCELPORT_VERBS_WITH_PERFORMANCE_COUNTERS)
  hpx_add_config_define_namespace(
      DEFINE    HPX_PARCELPORT_VERBS_HAVE_PERFORMANCE_COUNTERS
      NAMESPACE parcelport_verbs)
endif()

#------------------------------------------------------------------------------
# Throttling options
#------------------------------------------------------------------------------
hpx_option(HPX_PARCELPORT_VERBS_THROTTLE_SENDS STRING
  "Threshold of active sends at which throttling is enabled (default: 16)"
  "16" CATEGORY "Parcelport" ADVANCED)

hpx_add_config_define_namespace(
    DEFINE    HPX_PARCELPORT_VERBS_THROTTLE_SENDS
    VALUE     ${HPX_PARCELPORT_VERBS_THROTTLE_SENDS}
    NAMESPACE parcelport_verbs)

#------------------------------------------------------------------------------
# Custom Scheduler options
#------------------------------------------------------------------------------
hpx_option(HPX_PARCELPORT_VERBS_USE_CUSTOM_SCHEDULER BOOL
  "Configure the parcelport to use a custom scheduler "
  "(default: OFF - Warning, experimental, may cause serious program errors)"
  OFF CATEGORY "Parcelport" ADVANCED)

if (HPX_PARCELPORT_VERBS_USE_CUSTOM_SCHEDULER)
  hpx_add_config_define_namespace(
      DEFINE    HPX_PARCELPORT_VERBS_USE_CUSTOM_SCHEDULER
      NAMESPACE parcelport_verbs)
endif()

#------------------------------------------------------------------------------
# Lock checking
#------------------------------------------------------------------------------
hpx_option(HPX_PARCELPORT_VERBS_DEBUG_LOCKS BOOL
  "Turn on extra log messages for lock/unlock "
  "(default: OFF - Warning, severely impacts performance when enabled)"
  OFF CATEGORY "Parcelport" ADVANCED)

if (HPX_PARCELPORT_VERBS_DEBUG_LOCKS)
  hpx_add_config_define_namespace(
      DEFINE    HPX_PARCELPORT_VERBS_DEBUG_LOCKS
      NAMESPACE parcelport_verbs)
endif()

#------------------------------------------------------------------------------
# Memory chunk/reservation options
#------------------------------------------------------------------------------
hpx_option(HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE STRING
  "Number of bytes a default chunk in the memory pool can hold (default: 1K)"
  "1024" CATEGORY "Parcelport" ADVANCED)

hpx_option(HPX_PARCELPORT_VERBS_64K_PAGES STRING
  "Number of 64K pages we reserve for default message buffers (default: 10)"
  "10" CATEGORY "Parcelport" ADVANCED)

hpx_option(HPX_PARCELPORT_VERBS_MEMORY_COPY_THRESHOLD STRING
  "Cutoff size over which data is never copied into existing buffers (default: 4K)"
  "4096" CATEGORY "Parcelport" ADVANCED)

hpx_add_config_define_namespace(
    DEFINE    HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE
    VALUE     ${HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE}
    NAMESPACE parcelport_verbs)

# define the message header size to be equal to the chunk size
hpx_add_config_define_namespace(
    DEFINE    HPX_PARCELPORT_VERBS_MESSAGE_HEADER_SIZE
    VALUE     ${HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE}
    NAMESPACE parcelport_verbs)

hpx_add_config_define_namespace(
    DEFINE    HPX_PARCELPORT_VERBS_64K_PAGES
    VALUE     ${HPX_PARCELPORT_VERBS_64K_PAGES}
    NAMESPACE parcelport_verbs)

hpx_add_config_define_namespace(
    DEFINE    HPX_PARCELPORT_VERBS_MEMORY_COPY_THRESHOLD
    VALUE     ${HPX_PARCELPORT_VERBS_MEMORY_COPY_THRESHOLD}
    NAMESPACE parcelport_verbs)

#------------------------------------------------------------------------------
# Preposting options
#------------------------------------------------------------------------------
hpx_option(HPX_PARCELPORT_VERBS_MAX_PREPOSTS STRING
  "The number of pre-posted receive buffers (default: 512)"
  "512" CATEGORY "Parcelport" ADVANCED)

hpx_add_config_define_namespace(
    DEFINE    HPX_PARCELPORT_VERBS_MAX_PREPOSTS
    VALUE     ${HPX_PARCELPORT_VERBS_MAX_PREPOSTS}
    NAMESPACE parcelport_verbs)

#------------------------------------------------------------------------------
# Hardware checks
#------------------------------------------------------------------------------
if (HPX_PARCELPORT_VERBS_INTERFACE MATCHES "roq")
  message(WARNING "The roq interface on BGQ IO nodes does not support immediate data, disabling it")
  hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_VERBS_IMM_UNSUPPORTED
      VALUE  1
      NAMESPACE parcelport_verbs)
endif()

#------------------------------------------------------------------------------
# Write parcelport_verbs_defines.hpp
#------------------------------------------------------------------------------
# write config to build dir
write_config_defines_file(
  TEMPLATE ""
  NAMESPACE "parcelport_verbs"
  FILENAME "${CMAKE_BINARY_DIR}/hpx/config/parcelport_verbs_defines.hpp")

# write config to install dir
write_config_defines_file(
  TEMPLATE ""
  NAMESPACE "parcelport_verbs"
  FILENAME
  "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpx/config/parcelport_verbs_defines.hpp")

#------------------------------------------------------------------------------
# Define the ParcelPort registration macro
#------------------------------------------------------------------------------
include(HPX_AddLibrary)

# make these variables cache ones so that the macro definitions picks them up correctly
set(_verbs_libraries
    "${IB_VERBS_LIBRARIES}"
    "${RDMA_CM_LIBRARIES}"
    CACHE STRING "" FORCE)

set(_verbs_include_dirs
    "${IB_VERBS_INCLUDE_DIRS}"
    "${RDMA_CM_INCLUDE_DIRS}"
    CACHE STRING "" FORCE)

macro(add_parcelport_verbs_module)
  hpx_debug("add_parcelport_verbs_module")

  add_parcelport(verbs
    STATIC
    INCLUDE_DIRECTORIES
        ${_verbs_include_dirs}
    SOURCES
        "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/parcelport_verbs.cpp"
        "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/rdma/rdma_controller.cpp"
    # do we need to declare any of these headers?
    HEADERS
        "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/header.hpp"
        "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/locality.hpp"
        "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/pinned_memory_vector.hpp"
    DEPENDENCIES
        ${_verbs_libraries}
    FOLDER
      "Core/Plugins/Parcelport/verbs")
endmacro()
