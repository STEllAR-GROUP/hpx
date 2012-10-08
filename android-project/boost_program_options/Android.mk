
# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

MY_DIR:=$(call my-dir)
include $(MY_DIR)/../boost_common.mk

include $(CLEAR_VARS)
LOCAL_MODULE:=boost_program_options
LOCAL_CPP_EXTENSION:=.cpp
LOCAL_C_INCLUDES:=$(BOOST_SRC_ROOT)
#LOCAL_EXPORT_C_INCLUDES:=$(LOCAL_C_INCLUDES)
LOCAL_PATH:=$(BOOST_SRC_ROOT)
LOCAL_SRC_FILES:=libs/program_options/src/cmdline.cpp
LOCAL_SRC_FILES+=libs/program_options/src/config_file.cpp
LOCAL_SRC_FILES+=libs/program_options/src/options_description.cpp
LOCAL_SRC_FILES+=libs/program_options/src/parsers.cpp
LOCAL_SRC_FILES+=libs/program_options/src/variables_map.cpp
LOCAL_SRC_FILES+=libs/program_options/src/value_semantic.cpp
LOCAL_SRC_FILES+=libs/program_options/src/positional_options.cpp
LOCAL_SRC_FILES+=libs/program_options/src/utf8_codecvt_facet.cpp
LOCAL_SRC_FILES+=libs/program_options/src/convert.cpp
LOCAL_SRC_FILES+=libs/program_options/src/split.cpp
LOCAL_CPPFLAGS:=-DBOOST_PROGRAM_OPTIONS_DYN_LINK=1
NDK_TOOLCHAIN_VERSION:=4.6
LOCAL_ARM_NEON:=true
LOCAL_LDLIBS := -fuse-ld=gold
LOCAL_CPP_FEATURES:=exceptions rtti
include $(BUILD_SHARED_LIBRARY)
