# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_PROGRAM_OPTIONS_WITH_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
  if(NOT TARGET Boost::program_options)
    find_package(
      Boost ${Boost_MINIMUM_VERSION} MODULE COMPONENTS program_options
    )

    if(NOT Boost_PROGRAM_OPTIONS_FOUND)
      hpx_error(
        "Could not find Boost.ProgramOptions. Provide a boost installation including the program_options library"
      )
    endif()
  endif()

  set(__boost_program_options Boost::program_options)

  hpx_add_config_define_namespace(
    DEFINE HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY
    NAMESPACE PROGRAM_OPTIONS
  )
endif()
