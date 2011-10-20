# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2003 Jan Woetzel
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# cleans and removes cmake generated files etc.

if(UNIX)
  add_custom_target(distclean @echo Completed distclean, you may now re-run CMake.)

  add_custom_command(
    DEPENDS clean
    COMMAND rm
    ARGS    -f CMakeCache.txt
    TARGET  distclean)
endif()

