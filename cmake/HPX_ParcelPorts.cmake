# Copyright (c) 2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  hpx_debug("creating static_parcelports.hpp: "
            ${HPX_STATIC_PARCELPORT_PLUGINS}
  )

  # add_subdirectory is called before to insure HPX_STATIC_PARCELPORT_PLUGINS
  # cache variable is filled
  set(_parcelport_export)
  set(_parcelport_init)

  foreach(parcelport ${HPX_STATIC_PARCELPORT_PLUGINS})
    set(_parcelport_export
        "${_parcelport_export}HPX_EXPORT hpx::plugins::parcelport_factory_base *${parcelport}_factory_init(\n"
    )
    set(_parcelport_export
        "${_parcelport_export}    std::vector<hpx::plugins::parcelport_factory_base *>& factories);\n"
    )
    set(_parcelport_init
        "${_parcelport_init}        ${parcelport}_factory_init(factories);\n"
    )
  endforeach()

  configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/templates/static_parcelports.hpp.in"
    "${PROJECT_BINARY_DIR}/libs/full/parcelset/include/hpx/parcelset/static_parcelports.hpp"
    @ONLY
  )
endif()
