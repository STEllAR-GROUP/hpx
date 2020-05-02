# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2007-2019 Hartmut Kaiser
# Copyright (c) 2011-2014 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
# Copyright (c)      2013 Jeroen Habraken
# Copyright (c) 2014-2016 Andreas Schaefer
# Copyright (c) 2017      Abhimanyu Rawat
# Copyright (c) 2017      Google
# Copyright (c) 2017      Taeguk Kwon
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_HPXMP)

  # hpxMP requires ASM support
  enable_language(ASM)
  set(CMAKE_ASM_FLAGS "${CFLAGS} -x assembler-with-cpp")

  set(_hpxmp_no_update)
  if(HPX_WITH_HPXMP_NO_UPDATE)
    set(_hpxmp_no_update NO_UPDATE)
  endif()

  if(NOT HPXMP_ROOT)
    # handle hpxMP library
    include(GitExternal)
    git_external(
      hpxmp https://github.com/STEllAR-GROUP/hpxMP.git ${HPX_WITH_HPXMP_TAG}
      ${_hpxmp_no_update} VERBOSE
    )
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hpxmp)
      set(HPXMP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/hpxmp)
    else()
      hpx_error("hpxMP could not be found and HPX_WITH_HPXMP=On")
    endif()
  endif(NOT HPXMP_ROOT)

  add_subdirectory(${HPXMP_ROOT})

  # make sure thread-local storage is supported
  hpx_add_config_define(HPX_HAVE_THREAD_LOCAL_STORAGE)
endif()
