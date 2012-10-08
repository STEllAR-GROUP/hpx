
# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

MY_DIR:=$(call my-dir)
include $(MY_DIR)/../boost_common.mk

include $(CLEAR_VARS)
LOCAL_MODULE:=boost_context
LOCAL_CPP_EXTENSION:=.cpp
LOCAL_C_INCLUDES:=$(BOOST_SRC_ROOT)
#LOCAL_EXPORT_C_INCLUDES:=$(LOCAL_C_INCLUDES)
LOCAL_PATH:=$(BOOST_SRC_ROOT)
LOCAL_SRC_FILES:=libs/context/src/asm/make_arm_aapcs_elf_gas.S
LOCAL_SRC_FILES+=libs/context/src/asm/jump_arm_aapcs_elf_gas.S
LOCAL_SRC_FILES+=libs/context/src/fcontext.cpp
LOCAL_SRC_FILES+=libs/context/src/guarded_stack_allocator_posix.cpp
LOCAL_SRC_FILES+=libs/context/src/utils_posix.cpp
LOCAL_CPPFLAGS:=-DBOOST_CONTEXT_DYN_LINK=1
LOCAL_CPPFLAGS+=-std=gnu++0x
LOCAL_CPPFLAGS+=-DBOOST_SYSTEM_NO_DEPRECATED
LOCAL_CPPFLAGS+=-Wno-psabi
NDK_TOOLCHAIN_VERSION:=4.6
LOCAL_ARM_NEON:=true
LOCAL_LDLIBS := -fuse-ld=gold
LOCAL_CPP_FEATURES:=exceptions rtti
include $(BUILD_SHARED_LIBRARY)

