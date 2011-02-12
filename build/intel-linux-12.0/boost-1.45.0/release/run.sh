#! /usr/bin/env bash
#
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

ICC_VER=12.0
BOOST_VER=1.45.0

cmake -DCMAKE_C_COMPILER="/opt/intel/bin/icc"                       \
      -DCMAKE_CXX_COMPILER="/opt/intel/bin/icpc"                    \
      -DCMAKE_CXX_FLAGS:STRING="-ipo"                               \
      -DCMAKE_BUILD_TYPE=Release                                    \
      -DBOOST_DEBUG=ON                                              \
      -DBOOST_ROOT="/opt/boost-${BOOST_VER}"                        \
      -DBOOST_LIB_DIR="/opt/boost-${BOOST_VER}/intel-${ICC_VER}/lib"\
      -DCMAKE_PREFIX="."                                            \
      ../../../..

