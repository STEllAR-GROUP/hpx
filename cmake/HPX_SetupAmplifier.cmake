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
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if((NOT HPX_WITH_APEX) AND HPX_WITH_ITTNOTIFY)
  find_package(Amplifier)
  if(NOT AMPLIFIER_FOUND)
    hpx_error("Intel Amplifier could not be found and HPX_WITH_ITTNOTIFY=On, please specify AMPLIFIER_ROOT to point to the root of your Amplifier installation")
  endif()

  add_library(hpx::amplifier INTERFACE IMPORTED)
  target_include_directories(hpx::amplifier SYSTEM INTERFACE ${AMPLIFIER_INCLUDE_DIR})
  target_link_libraries(hpx::amplifier INTERFACE ${AMPLIFIER_LIBRARIES})

  hpx_add_config_define(HPX_HAVE_ITTNOTIFY 1)
  hpx_add_config_define(HPX_HAVE_THREAD_DESCRIPTION)
endif()

# convey selected allocator type to the build configuration
hpx_add_config_define(HPX_HAVE_MALLOC "\"${HPX_WITH_MALLOC}\"")
if(${HPX_WITH_MALLOC} STREQUAL "jemalloc")
  if(NOT ("${HPX_WITH_JEMALLOC_PREFIX}" STREQUAL "<none>") AND
     NOT ("${HPX_WITH_JEMALLOC_PREFIX}x" STREQUAL "x"))
    hpx_add_config_define(HPX_HAVE_JEMALLOC_PREFIX ${HPX_WITH_JEMALLOC_PREFIX})
    hpx_add_config_define(HPX_HAVE_INTERNAL_ALLOCATOR)
  endif()
endif()
