# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if((HPX_WITH_COMPRESSION_BZIP2
    OR HPX_WITH_COMPRESSION_SNAPPY
    OR HPX_WITH_COMPRESSION_ZLIB)
   AND NOT TARGET Boost::iostreams
)
  find_package(Boost ${Boost_MINIMUM_VERSION} MODULE COMPONENTS iostreams)

  if(Boost_IOSTREAMS_FOUND)
    hpx_info("  iostreams")
  else()
    hpx_error(
      "Could not find Boost.Iostreams but HPX_WITH_COMPRESSION_BZIP2=On or \
    HPX_WITH_COMPRESSION_LIB=On. Either set it to off or provide a boost installation including \
    the iostreams library"
    )
  endif()
endif()
