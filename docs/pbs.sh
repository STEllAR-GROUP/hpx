#! /bin/bash
#
# Copyright (c) 2009-2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file BOOST_LICENSE_1_0.rst or copy at http://www.boost.org/LICENSE_1_0.txt)

APP_PATH    =
APP_OPTIONS =

pbsdsh -u $APP_PATH --nodes=`echo $PBS_NODEFILE` $APP_OPTIONS

