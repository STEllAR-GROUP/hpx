
# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

MY_DIR:=$(call my-dir)
include $(MY_DIR)/../hpx_common.mk

include $(CLEAR_VARS)
LOCAL_MODULE:=hpx
LOCAL_CPP_EXTENSION:=.cpp
LOCAL_C_INCLUDES:=$(LOCAL_PATH)/boost
LOCAL_C_INCLUDES+=$(HPX_INCLUDES)
LOCAL_PATH:=$(HPX_SRC_ROOT)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/runtime/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/runtime/parcelset/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/runtime/threads/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/runtime/naming/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/runtime/actions/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/runtime/applier/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/runtime/agas/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/runtime/agas/stubs/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/runtime/agas/server/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/runtime/components/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/runtime/components/server/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/performance_counters/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/performance_counters/stubs/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/performance_counters/server/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/util/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/lcos/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/lcos/local/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/lcos/barrier/*.cpp)
LOCAL_SRC_FILES+=$(wildcard $(HPX_SRC_ROOT)/src/lcos/detail/*.cpp)
LOCAL_SRC_FILES:=$(filter-out $(HPX_SRC_ROOT)/src/main.cpp, $(LOCAL_SRC_FILES))
LOCAL_SRC_FILES:=$(filter-out $(HPX_SRC_ROOT)/src/hpx_main_argc_argv.cpp, $(LOCAL_SRC_FILES))
LOCAL_SRC_FILES:=$(filter-out $(HPX_SRC_ROOT)/src/hpx_main.cpp, $(LOCAL_SRC_FILES))
LOCAL_SRC_FILES:=$(filter-out $(HPX_SRC_ROOT)/src/hpx_main_variables_map.cpp, $(LOCAL_SRC_FILES))
LOCAL_SRC_FILES:=$(filter-out $(HPX_SRC_ROOT)/src/hpx_user_main_argc_argv.cpp, $(LOCAL_SRC_FILES))
LOCAL_SRC_FILES:=$(filter-out $(HPX_SRC_ROOT)/src/hpx_user_main.cpp, $(LOCAL_SRC_FILES))
LOCAL_SRC_FILES:=$(filter-out $(HPX_SRC_ROOT)/src/util/portable_binary_iarchive.cpp, $(LOCAL_SRC_FILES))
LOCAL_SRC_FILES:=$(filter-out $(HPX_SRC_ROOT)/src/util/portable_binary_oarchive.cpp, $(LOCAL_SRC_FILES))
LOCAL_SRC_FILES:=$(patsubst $(HPX_SRC_ROOT)/%, %, $(LOCAL_SRC_FILES))
LOCAL_CPPFLAGS:=$(HPX_CPPFLAGS)
LOCAL_CPPFLAGS+=-DHPX_LIBRARY=\"hpx\"
LOCAL_CPPFLAGS+=-DHPX_EXPORTS
LOCAL_CPPFLAGS+=-DHPX_COROUTINE_EXPORTS
LOCAL_CPPFLAGS+=-DHPX_ACTION_ARGUMENT_LIMIT=4
LOCAL_CPPFLAGS+=-DHPX_FUNCTION_ARGUMENT_LIMIT=7
LOCAL_CPPFLAGS+=-DBOOST_ENABLE_ASSERT_HANDLER
LOCAL_CPPFLAGS+=-DPAGE_SIZE="(1UL << 12)"

LOCAL_LDLIBS := -fuse-ld=gold
LOCAL_STATIC_LIBRARIES := cpufeatures
LOCAL_SHARED_LIBRARIES := boost_system boost_thread boost_serialization boost_chrono boost_atomic hpx_serialization boost_context boost_regex boost_date_time boost_program_options boost_filesystem
NDK_TOOLCHAIN_VERSION:=4.6
LOCAL_ARM_NEON:=true
LOCAL_CPP_FEATURES:=exceptions rtti
include $(BUILD_SHARED_LIBRARY)

$(call import-module, boost_system)
$(call import-module, boost_thread)
$(call import-module, boost_serialization)
$(call import-module, boost_chrono)
$(call import-module, boost_atomic)
$(call import-module, boost_context)
$(call import-module, boost_regex)
$(call import-module, boost_date_time)
$(call import-module, boost_program_options)
$(call import-module, boost_filesystem)
$(call import-module, hpx_serialization)
$(call import-module, hpx_init)
$(call import-module, cpufeatures)
