
# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

MY_DIR:=$(call my-dir)
include $(MY_DIR)/../boost_common.mk

include $(CLEAR_VARS)
LOCAL_MODULE:=boost_system
LOCAL_CPP_EXTENSTION:=.cpp

LOCAL_PATH:=$(BOOST_SRC_ROOT)
LOCAL_SRC_FILES:=libs/system/src/error_code.cpp

LOCAL_C_INCLUDES:=$(BOOST_SRC_ROOT)
LOCAL_EXPORT_C_INCLUDES:=$(LOCAL_C_INCLUDES)


LOCAL_CPPFLAGS:=-DBOOST_SYSTEM_DYN_LINK=1

LOCAL_ARM_NEON:=true

LOCAL_LDLIBS := -fuse-ld=gold

LOCAL_CPP_FEATURES:=exceptions rtti
include $(BUILD_SHARED_LIBRARY)

