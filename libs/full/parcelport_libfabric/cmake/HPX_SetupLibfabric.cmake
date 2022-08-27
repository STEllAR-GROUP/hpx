# Copyright (c) 2019 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# FIXME : in the future put it directly inside the cmake directory of the
# corresponding plugin

if(HPX_WITH_PARCELPORT_LIBFABRIC AND NOT TARGET Libfabric::libfabric)
  # ------------------------------------------------------------------------------
  # Add #define to global defines.hpp
  # ------------------------------------------------------------------------------
  hpx_add_config_define(HPX_HAVE_PARCELPORT_LIBFABRIC)

  # ------------------------------------------------------------------------------
  # OFIWG libfabric stack
  # ------------------------------------------------------------------------------
  find_package(Libfabric REQUIRED)

  # Setup Libfabric imported target
  add_library(Libfabric::libfabric INTERFACE IMPORTED)
  target_include_directories(
    Libfabric::libfabric SYSTEM INTERFACE ${LIBFABRIC_INCLUDE_DIR}
  )
  target_link_libraries(Libfabric::libfabric INTERFACE ${LIBFABRIC_LIBRARY})

  # Setup PMI imported target
  find_package(PMI)
  if(PMI_FOUND)
    hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_LIBFABRIC_HAVE_PMI NAMESPACE PARCELPORT_LIBFABRIC
    )
    add_library(Pmi::pmi INTERFACE IMPORTED)
    target_include_directories(Pmi::pmi SYSTEM INTERFACE ${PMI_INCLUDE_DIR})
    target_link_libraries(Pmi::pmi INTERFACE ${PMI_LIBRARY})
  endif()

  # ------------------------------------------------------------------------------
  # Logging
  # ------------------------------------------------------------------------------
  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_WITH_LOGGING
    BOOL
    "Enable logging in the libfabric ParcelPort (default: OFF - Warning - severely impacts usability when enabled)"
    OFF
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  if(HPX_PARCELPORT_LIBFABRIC_WITH_LOGGING)
    hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_LIBFABRIC_HAVE_LOGGING
      NAMESPACE PARCELPORT_LIBFABRIC
    )
  endif()

  # ------------------------------------------------------------------------------
  # Development mode (extra logging and customizable tweaks)
  # ------------------------------------------------------------------------------
  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_WITH_DEV_MODE
    BOOL
    "Enables some extra logging and debug features in the libfabric parcelport (default: OFF - Warning - severely impacts usability when enabled)"
    OFF
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  if(HPX_PARCELPORT_LIBFABRIC_WITH_DEV_MODE)
    hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_LIBFABRIC_HAVE_DEV_MODE
      NAMESPACE PARCELPORT_LIBFABRIC
    )
  endif()

  # ------------------------------------------------------------------------------
  # make sure boost log is linked correctly
  # ------------------------------------------------------------------------------
  if(HPX_PARCELPORT_LIBFABRIC_WITH_LOGGING
     OR HPX_PARCELPORT_LIBFABRIC_WITH_DEV_MODE
  )
    if(NOT Boost_USE_STATIC_LIBS)
      hpx_add_config_define_namespace(
        DEFINE BOOST_LOG_DYN_LINK NAMESPACE PARCELPORT_LIBFABRIC
      )
    endif()
  endif()

  # ------------------------------------------------------------------------------
  # Hardware device selection
  # ------------------------------------------------------------------------------
  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_PROVIDER STRING
    "The provider (verbs/gni/psm2/sockets)" "verbs"
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  hpx_add_config_define_namespace(
    DEFINE HPX_PARCELPORT_LIBFABRIC_PROVIDER
    VALUE "\"${HPX_PARCELPORT_LIBFABRIC_PROVIDER}\""
    NAMESPACE PARCELPORT_LIBFABRIC
  )

  if(HPX_PARCELPORT_LIBFABRIC_PROVIDER MATCHES "verbs")
    hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_LIBFABRIC_VERBS NAMESPACE PARCELPORT_LIBFABRIC
    )
  elseif(HPX_PARCELPORT_LIBFABRIC_PROVIDER MATCHES "gni")
    hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_LIBFABRIC_GNI NAMESPACE PARCELPORT_LIBFABRIC
    )
  elseif(HPX_PARCELPORT_LIBFABRIC_PROVIDER MATCHES "sockets")
    hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_LIBFABRIC_SOCKETS NAMESPACE PARCELPORT_LIBFABRIC
    )
  elseif(HPX_PARCELPORT_LIBFABRIC_PROVIDER MATCHES "psm2")
    hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_LIBFABRIC_PSM2 NAMESPACE PARCELPORT_LIBFABRIC
    )
  endif()

  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_DOMAIN STRING
    "The libfabric domain (leave blank for default" ""
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  hpx_add_config_define_namespace(
    DEFINE HPX_PARCELPORT_LIBFABRIC_DOMAIN
    VALUE "\"${HPX_PARCELPORT_LIBFABRIC_DOMAIN}\""
    NAMESPACE PARCELPORT_LIBFABRIC
  )

  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_ENDPOINT STRING
    "The libfabric endpoint type (leave blank for default" "rdm"
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  hpx_add_config_define_namespace(
    DEFINE HPX_PARCELPORT_LIBFABRIC_ENDPOINT
    VALUE "\"${HPX_PARCELPORT_LIBFABRIC_ENDPOINT}\""
    NAMESPACE PARCELPORT_LIBFABRIC
  )

  # ------------------------------------------------------------------------------
  # Bootstrap options
  # ------------------------------------------------------------------------------
  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_WITH_BOOTSTRAPPING
    BOOL
    "Configure the parcelport to enable bootstrap capabilities (default: OFF, enabled if PMI was found)"
    ${PMI_FOUND}
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  if(HPX_PARCELPORT_LIBFABRIC_WITH_BOOTSTRAPPING)
    if(NOT PMI_FOUND)
      hpx_error(
        "For libfabric bootstrapping to work, the PMI libraries must be enabled and found"
      )
    endif()

    hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_LIBFABRIC_HAVE_BOOTSTRAPPING
      VALUE std::true_type
      NAMESPACE PARCELPORT_LIBFABRIC
    )
  else()
    hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_LIBFABRIC_HAVE_BOOTSTRAPPING
      VALUE std::false_type
      NAMESPACE PARCELPORT_LIBFABRIC
    )
  endif()

  # ------------------------------------------------------------------------------
  # Performance counters
  # ------------------------------------------------------------------------------
  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_WITH_PERFORMANCE_COUNTERS BOOL
    "Enable libfabric parcelport performance counters (default: OFF)" OFF
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  if(HPX_PARCELPORT_LIBFABRIC_WITH_PERFORMANCE_COUNTERS)
    hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_LIBFABRIC_HAVE_PERFORMANCE_COUNTERS
      NAMESPACE PARCELPORT_LIBFABRIC
    )
  endif()

  # ------------------------------------------------------------------------------
  # Throttling options
  # ------------------------------------------------------------------------------
  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_THROTTLE_SENDS STRING
    "Threshold of active sends at which throttling is enabled (default: 16)"
    "16"
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  hpx_add_config_define_namespace(
    DEFINE HPX_PARCELPORT_LIBFABRIC_THROTTLE_SENDS
    VALUE ${HPX_PARCELPORT_LIBFABRIC_THROTTLE_SENDS}
    NAMESPACE PARCELPORT_LIBFABRIC
  )

  # ------------------------------------------------------------------------------
  # Custom Scheduler options
  # ------------------------------------------------------------------------------
  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_USE_CUSTOM_SCHEDULER
    BOOL
    "Configure the parcelport to use a custom scheduler (default: OFF - Warning, experimental, may cause serious program errors)"
    OFF
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  if(HPX_PARCELPORT_LIBFABRIC_USE_CUSTOM_SCHEDULER)
    hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_LIBFABRIC_USE_CUSTOM_SCHEDULER
      NAMESPACE PARCELPORT_LIBFABRIC
    )
  endif()

  # ------------------------------------------------------------------------------
  # Lock checking
  # ------------------------------------------------------------------------------
  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_DEBUG_LOCKS
    BOOL
    "Turn on extra log messages for lock/unlock  (default: OFF - Warning, severely impacts performance when enabled)"
    OFF
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  if(HPX_PARCELPORT_LIBFABRIC_DEBUG_LOCKS)
    hpx_add_config_define_namespace(
      DEFINE HPX_PARCELPORT_LIBFABRIC_DEBUG_LOCKS
      NAMESPACE PARCELPORT_LIBFABRIC
    )
  endif()

  # ------------------------------------------------------------------------------
  # Memory chunk/reservation options
  # ------------------------------------------------------------------------------
  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_MEMORY_CHUNK_SIZE STRING
    "Number of bytes a default chunk in the memory pool can hold (default: 4K)"
    "4096"
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_64K_PAGES STRING
    "Number of 64K pages we reserve for default message buffers (default: 10)"
    "10"
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_MEMORY_COPY_THRESHOLD
    STRING
    "Cutoff size over which data is never copied into existing buffers (default: 4K)"
    "4096"
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  hpx_add_config_define_namespace(
    DEFINE HPX_PARCELPORT_LIBFABRIC_MEMORY_CHUNK_SIZE
    VALUE ${HPX_PARCELPORT_LIBFABRIC_MEMORY_CHUNK_SIZE}
    NAMESPACE PARCELPORT_LIBFABRIC
  )

  # define the message header size to be equal to the chunk size
  hpx_add_config_define_namespace(
    DEFINE HPX_PARCELPORT_LIBFABRIC_MESSAGE_HEADER_SIZE
    VALUE ${HPX_PARCELPORT_LIBFABRIC_MEMORY_CHUNK_SIZE}
    NAMESPACE PARCELPORT_LIBFABRIC
  )

  hpx_add_config_define_namespace(
    DEFINE HPX_PARCELPORT_LIBFABRIC_64K_PAGES
    VALUE ${HPX_PARCELPORT_LIBFABRIC_64K_PAGES}
    NAMESPACE PARCELPORT_LIBFABRIC
  )

  hpx_add_config_define_namespace(
    DEFINE HPX_PARCELPORT_LIBFABRIC_MEMORY_COPY_THRESHOLD
    VALUE ${HPX_PARCELPORT_LIBFABRIC_MEMORY_COPY_THRESHOLD}
    NAMESPACE PARCELPORT_LIBFABRIC
  )

  # ------------------------------------------------------------------------------
  # Preposting options
  # ------------------------------------------------------------------------------
  hpx_option(
    HPX_PARCELPORT_LIBFABRIC_MAX_PREPOSTS STRING
    "The number of pre-posted receive buffers (default: 512)" "512"
    CATEGORY "Parcelport"
    ADVANCED
    MODULE PARCELPORT_LIBFABRIC
  )

  hpx_add_config_define_namespace(
    DEFINE HPX_PARCELPORT_LIBFABRIC_MAX_PREPOSTS
    VALUE ${HPX_PARCELPORT_LIBFABRIC_MAX_PREPOSTS}
    NAMESPACE PARCELPORT_LIBFABRIC
  )

endif()
