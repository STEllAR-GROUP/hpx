#!/bin/bash

# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Out-of-source build

EXPECTED_ARGS=2

if [ $# -ne $EXPECTED_ARGS ]
then
  echo "Usage: build.sh <pxgl-source> <pxgl-install>"
  exit -1
fi

PXGL_SOURCE=$1
PXGL_INSTALL=$2

BOOST_INSTALL=$HOME/usr/local/boost_1_43_0
HPX_INSTALL=$HOME/usr/local/hpx_svn_relwithdebinfo

echo "cmake $PXGL_SOURCE -DCMAKE_INSTALL_PREFIX=$PXGL_INSTALL -DBOOST_ROOT=$BOOST_INSTALL -DHPX_ROOT=$HPX_INSTALL"
