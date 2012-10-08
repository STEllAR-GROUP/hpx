
# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

BOOST_SRC_ROOT:=$(strip $(BOOST_SRC_ROOT))
#/home/heller/programming/boost/trunk

ifndef BOOST_SRC_ROOT
  $(call __ndk_info,ERROR: You BOOST_SRC_ROOT not set)
  $(call __ndk_info,Please set BOOST_SRC_ROOT to point to your Boost source directory and start again.)
  $(call __ndk_error,Aborting)
endif
