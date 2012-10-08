
# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

MY_DIR:=$(call my-dir)
include $(MY_DIR)/../boost_common.mk

include $(CLEAR_VARS)
LOCAL_MODULE:=boost_thread
LOCAL_CPP_EXTENSTION:=.cpp

LOCAL_PATH:=$(BOOST_SRC_ROOT)
LOCAL_SRC_FILES:=libs/thread/src/pthread/thread.cpp
LOCAL_SRC_FILES+=libs/thread/src/pthread/once.cpp
LOCAL_SRC_FILES+=libs/thread/src/future.cpp 

LOCAL_CPPFLAGS:=-DBOOST_THREAD_BUILD_DLL=1
LOCAL_CPPFLAGS+=-DBOOST_THREAD_THROW_IF_PRECONDITION_NOT_SATISFIED
LOCAL_CPPFLAGS+=-DBOOST_THREAD_POSIX
LOCAL_CPPFLAGS+=-DBOOST_THREAD_DONT_USE_CHRONO
# this was taken from android-ndk-r8b/platforms/android-14/arch-arm/usr/include/asm/page.h"
LOCAL_CPPFLAGS+=-DPAGE_SIZE="(1UL << 12)"

LOCAL_SHARED_LIBRARIES:=boost_system
NDK_TOOLCHAIN_VERSION:=4.6
LOCAL_ARM_NEON:=true
LOCAL_LDLIBS := -fuse-ld=gold
LOCAL_CPP_FEATURES:=exceptions rtti
include $(BUILD_SHARED_LIBRARY)

$(call import-module,boost_system)
