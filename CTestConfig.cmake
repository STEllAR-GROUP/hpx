# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
# This file should be placed in the root directory of your project. Then modify
# the CMakeLists.txt file in the root directory of your project to incorporate
# the testing dashboard.
# The following are required to uses Dart and the dashboard
#    ENABLE_TESTING()
#    INCLUDE(CTest)
#
set(CTEST_PROJECT_NAME "HPX")
set(CTEST_NIGHTLY_START_TIME "00:00:00 GMT")

set(CTEST_DROP_METHOD "https")
set(CTEST_DROP_SITE "cdash.cscs.ch")
set(CTEST_DROP_LOCATION "/submit.php?project=HPX")
set(CTEST_DROP_SITE_CDASH TRUE)
