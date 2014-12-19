# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(parcelport_plugins
  ibverbs
  ipc
  mpi
  tcp)

foreach(type ${parcelport_plugins})
  add_hpx_pseudo_target(plugins.parcelport.${type})
  add_subdirectory(${type})
  add_hpx_pseudo_dependencies(plugins.parcelport plugins.parcelport.${type})
endforeach()

macro(add_parcelport_modules)
  add_parcelport_ibverbs_module()
  add_parcelport_ipc_module()
  add_parcelport_mpi_module()
  add_parcelport_tcp_module()
endmacro()
