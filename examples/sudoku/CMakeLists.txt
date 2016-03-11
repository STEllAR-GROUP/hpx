# Copyright (c) 2016 Satyaki Upadhyay
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_component(sudoku
  SOURCE_GLOB "sudoku.cp*"
  HEADER_GLOB "sudoku.hp*"
  DEPENDENCIES iostreams_component
  FOLDER "Examples/Applications/sudoku"
  AUTOGLOB)

add_hpx_executable(sudoku_client
  SOURCES sudoku_client.cpp
  DEPENDENCIES sudoku_component iostreams_component
  FOLDER "Examples/Applications/sudoku")

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.sudoku sudoku_client_exe)

