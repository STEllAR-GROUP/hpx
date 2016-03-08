# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2011 Vinay C Amatya
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
find_package(HPX)

add_hpx_component(nqueen
  SOURCE_GLOB "nqueen.cp*"
  HEADER_GLOB "nqueen.hp*"
  FOLDER "Examples/Applications/NQueen"
  AUTOGLOB)

add_hpx_executable(nqueen_client
  SOURCES nqueen_client.cpp
  DEPENDENCIES nqueen_component
  FOLDER "Examples/Applications/NQueen")

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.nqueen nqueen_client_exe)

