
# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

MY_DIR:=$(call my-dir)
include $(MY_DIR)/../boost_common.mk

include $(CLEAR_VARS)
LOCAL_MODULE:=boost_serialization
LOCAL_CPP_EXTENSION:=.cpp
LOCAL_C_INCLUDES:=$(BOOST_SRC_ROOT)
#LOCAL_EXPORT_C_INCLUDES:=$(LOCAL_C_INCLUDES)
LOCAL_PATH:=$(BOOST_SRC_ROOT)
LOCAL_SRC_FILES:=libs/serialization/src/basic_archive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/basic_iarchive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/basic_iserializer.cpp
LOCAL_SRC_FILES+=libs/serialization/src/basic_oarchive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/basic_oserializer.cpp
LOCAL_SRC_FILES+=libs/serialization/src/basic_pointer_iserializer.cpp
LOCAL_SRC_FILES+=libs/serialization/src/basic_pointer_oserializer.cpp
LOCAL_SRC_FILES+=libs/serialization/src/basic_serializer_map.cpp
LOCAL_SRC_FILES+=libs/serialization/src/basic_text_iprimitive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/basic_text_oprimitive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/basic_xml_archive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/binary_iarchive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/binary_oarchive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/extended_type_info.cpp
LOCAL_SRC_FILES+=libs/serialization/src/extended_type_info_typeid.cpp
LOCAL_SRC_FILES+=libs/serialization/src/extended_type_info_no_rtti.cpp
LOCAL_SRC_FILES+=libs/serialization/src/polymorphic_iarchive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/polymorphic_oarchive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/stl_port.cpp
LOCAL_SRC_FILES+=libs/serialization/src/text_iarchive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/text_oarchive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/void_cast.cpp
LOCAL_SRC_FILES+=libs/serialization/src/archive_exception.cpp
LOCAL_SRC_FILES+=libs/serialization/src/xml_grammar.cpp
LOCAL_SRC_FILES+=libs/serialization/src/xml_iarchive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/xml_oarchive.cpp
LOCAL_SRC_FILES+=libs/serialization/src/xml_archive_exception.cpp
LOCAL_SRC_FILES+=libs/serialization/src/shared_ptr_helper.cpp
#LOCAL_SRC_FILES+=stdlib.cpp
LOCAL_CPPFLAGS:=-DBOOST_SERIALIZATION_DYN_LINK=1
LOCAL_CPPFLAGS+=-DBOOST_ALL_NO_LIB=1
LOCAL_CPPFLAGS+=-DBOOST_NO_CWCHAR=1
NDK_TOOLCHAIN_VERSION:=4.6
LOCAL_ARM_NEON:=true
LOCAL_LDLIBS := -fuse-ld=gold
LOCAL_CPP_FEATURES:=exceptions rtti
include $(BUILD_SHARED_LIBRARY)

