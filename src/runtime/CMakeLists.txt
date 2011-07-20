# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(actions)

if("${HPX_AGAS_VERSION}" STREQUAL "2")
  add_subdirectory(agas)
endif()

add_subdirectory(applier)
add_subdirectory(components)
add_subdirectory(naming)
add_subdirectory(parcelset)
add_subdirectory(threads)

