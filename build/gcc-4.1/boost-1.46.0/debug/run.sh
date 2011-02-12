#! /usr/bin/env bash
#
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

GCC_VER=4.1
BOOST_VER=1.46.0

cmake -DCMAKE_C_COMPILER="/usr/bin/gcc-${GCC_VER}"                  \
      -DCMAKE_CXX_COMPILER="/usr/bin/g++-${GCC_VER}"                \
      -DCMAKE_BUILD_TYPE=Debug                                      \
      -DBOOST_DEBUG=ON                                              \
      -DBOOST_ROOT="/opt/boost-${BOOST_VER}"                        \
      -DBOOST_LIB_DIR="/opt/boost-${BOOST_VER}/gcc-${GCC_VER}/lib"  \
      -DCMAKE_PREFIX="."                                            \
      ../../../..

