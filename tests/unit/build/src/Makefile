# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

.PHONY: all

ifeq ($(BUILD_TYPE),Debug)
  hpx_application = hpx_application_debug
  hpx_component = hpx_component_debug
  debug_suffix = d
else
  hpx_application = hpx_application
  hpx_component = hpx_component
  debug_suffix =
endif

APP_CFLAGS = $(CXX_FLAGS)
APP_CFLAGS += $(shell PKG_CONFIG_PATH=$(HPX_DIR) $(PKG_CONFIG) --cflags --libs $(hpx_application))
APP_CFLAGS += -lhpx_iostreams$(debug_suffix)
APP_CFLAGS += -DHPX_APPLICATION_NAME=test

COMP_CFLAGS = $(CXX_FLAGS)
COMP_CFLAGS += $(shell PKG_CONFIG_PATH=$(HPX_DIR) $(PKG_CONFIG) --cflags --libs $(hpx_component))
COMP_CFLAGS += -DHPX_COMPONENT_NAME=test_component

ifneq ($(wildcard $(HPX_DIR)/*),)
all: test
else
all:
	@echo "HPX_DIR=$(HPX_DIR) does not exist. Did you forget to run the install rule?"
endif

test: $(SRC_DIR)/test.cpp libtest_component.so
	$(CXX) -o test -I$(SRC_DIR)/ $(SRC_DIR)/test.cpp $(APP_CFLAGS) libtest_component.so

libtest_component.so: $(SRC_DIR)/test_component.cpp
	$(CXX) -o libtest_component.so -fPIC -I$(SRC_DIR)/ $(SRC_DIR)/test_component.cpp $(COMP_CFLAGS)
