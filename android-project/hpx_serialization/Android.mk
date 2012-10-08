
# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

MY_DIR:=$(call my-dir)
include $(MY_DIR)/../hpx_common.mk

include $(CLEAR_VARS)
LOCAL_MODULE:=hpx_serialization
LOCAL_CPP_EXTENSION:=.cpp
LOCAL_C_INCLUDES:=$(LOCAL_PATH)/boost
LOCAL_C_INCLUDES+=$(HPX_INCLUDES)
LOCAL_PATH:=$(HPX_SRC_ROOT)
LOCAL_SRC_FILES:=$(wildcard $(HPX_SRC_ROOT)/src/util/portable_binary_*archive.cpp)
LOCAL_SRC_FILES:=$(patsubst $(HPX_SRC_ROOT)/%, %, $(LOCAL_SRC_FILES))
LOCAL_CPPFLAGS:=$(HPX_CPPFLAGS)
LOCAL_CPPFLAGS+=-DHPX_LIBRARY=\"hpx_serialization\"
LOCAL_CPPFLAGS+=-DHPX_EXPORTS
LOCAL_CPPFLAGS+=-DHPX_COROUTINE_EXPORTS
#LOCAL_CPPFLAGS+=-DPAGE_SIZE="(1UL << 12)"
LOCAL_LDLIBS := -fuse-ld=gold
LOCAL_STATIC_LIBRARIES := cpufeatures
LOCAL_SHARED_LIBRARIES := boost_system boost_serialization
NDK_TOOLCHAIN_VERSION:=4.6
LOCAL_ARM_NEON:=true
LOCAL_CPP_FEATURES:=exceptions rtti
include $(BUILD_SHARED_LIBRARY)

$(call import-module, boost_system)
$(call import-module, boost_serialization)
