# Copyright (c) 2021 ETH Zurich
# Copyright (c) 2024 The STE||AR Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

configure_extra_options+=" -DCMAKE_BUILD_TYPE=${build_type}"
configure_extra_options+=" -DHPX_WITH_CHECK_MODULE_DEPENDENCIES=ON"
if [ "${build_type}" = "Debug" ]; then
    configure_extra_options+=" -DHPX_WITH_PARCELPORT_COUNTERS=ON"
    configure_extra_options+=" -DLCI_DEBUG=ON"
    configure_extra_options+=" -DHPX_WITH_VERIFY_LOCKS=ON"
#    configure_extra_options+=" -DHPX_WITH_VERIFY_LOCKS_BACKTRACE=ON"
fi

ctest_extra_args+=" --verbose "

hostname
module avail
