# Copyright (c) 2021 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

configure_extra_options+=" -DCMAKE_BUILD_TYPE=${build_type}"
configure_extra_options+=" -DHPXLocal_WITH_CHECK_MODULE_DEPENDENCIES=ON"
configure_extra_options+=" -DHPXLocal_WITH_EXAMPLES=ON"
configure_extra_options+=" -DHPXLocal_WITH_TESTS=ON"
configure_extra_options+=" -DHPXLocal_WITH_TESTS_UNIT=ON"
configure_extra_options+=" -DHPXLocal_WITH_TESTS_REGRESSIONS=ON"
configure_extra_options+=" -DHPXLocal_WITH_TESTS_BENCHMARKS=ON"
configure_extra_options+=" -DHPXLocal_WITH_TESTS_EXAMPLES=ON"
configure_extra_options+=" -DHPXLocal_WITH_TESTS_EXTERNAL_BUILD=ON"
