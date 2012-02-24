#! /bin/bash
#
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file BOOST_LICENSE_1_0.rst or copy at http://www.boost.org/LICENSE_1_0.txt)
#
#PBS -l nodes=N:ppn=M

APP_PATH=
APP_OPTIONS=

pbsdsh -u $APP_PATH --hpx:nodes=`cat $PBS_NODEFILE` $APP_OPTIONS

