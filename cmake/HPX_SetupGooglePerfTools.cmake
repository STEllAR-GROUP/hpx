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
# Copyright (c) 2018 Christopher Hinz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_GOOGLE_PERFTOOLS)
  find_package(GooglePerftools)
  if(NOT GOOGLE_PERFTOOLS_FOUND)
    hpx_error("Google Perftools could not be found and \
    HPX_WITH_GOOGLE_PERFTOOLS=On, please specify GOOGLE_PERFTOOLS to point to \
    the root of your Google Perftools installation")
  endif()

  add_library(hpx::gperftools INTERFACE IMPORTED)
  target_include_directories(hpx::gperftool SYSTEM INTERFACE ${GOOGLE_PERFTOOLS_INCLUDE_DIR})
  target_link_libraries(hpx::gperftools INTERFACE ${GOOGLE_PERFTOOLS_LIBRARIES})
endif()
