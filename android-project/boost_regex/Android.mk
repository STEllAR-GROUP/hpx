
# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

MY_DIR:=$(call my-dir)
include $(MY_DIR)/../boost_common.mk

include $(CLEAR_VARS)
LOCAL_MODULE:=boost_regex
LOCAL_CPP_EXTENSION:=.cpp
LOCAL_C_INCLUDES:=$(BOOST_SRC_ROOT)
#LOCAL_EXPORT_C_INCLUDES:=$(LOCAL_C_INCLUDES)
LOCAL_PATH:=$(BOOST_SRC_ROOT)
LOCAL_SRC_FILES:=libs/regex/src/c_regex_traits.cpp
LOCAL_SRC_FILES+=libs/regex/src/cpp_regex_traits.cpp
LOCAL_SRC_FILES+=libs/regex/src/cregex.cpp
LOCAL_SRC_FILES+=libs/regex/src/fileiter.cpp
LOCAL_SRC_FILES+=libs/regex/src/instances.cpp
LOCAL_SRC_FILES+=libs/regex/src/posix_api.cpp
LOCAL_SRC_FILES+=libs/regex/src/regex.cpp
LOCAL_SRC_FILES+=libs/regex/src/regex_debug.cpp
LOCAL_SRC_FILES+=libs/regex/src/regex_raw_buffer.cpp
LOCAL_SRC_FILES+=libs/regex/src/regex_traits_defaults.cpp
LOCAL_SRC_FILES+=libs/regex/src/static_mutex.cpp
LOCAL_SRC_FILES+=libs/regex/src/wc_regex_traits.cpp
LOCAL_SRC_FILES+=libs/regex/src/wide_posix_api.cpp
LOCAL_SRC_FILES+=libs/regex/src/winstances.cpp
LOCAL_SRC_FILES+=libs/regex/src/usinstances.cpp
LOCAL_CPPFLAGS:=-DBOOST_REGEX_DYN_LINK=1
NDK_TOOLCHAIN_VERSION:=4.6
LOCAL_ARM_NEON:=true
LOCAL_LDLIBS := -fuse-ld=gold
LOCAL_CPP_FEATURES:=exceptions rtti
include $(BUILD_SHARED_LIBRARY)

