# Copyright (c) 2018 Patrick Diehl
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(CPACK_PACKAGE_VERSION ${HPX_VERSION})
set(CPACK_GENERATOR "RPM")
set(CPACK_PACKAGE_NAME "hpx")
set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_CONTACT "contact@stellar-group.org")
set(CPACK_PACKAGE_VENDOR "STE||AR Group")
set(CPACK_RPM_PACKAGE_LICENSE "BSL-1.0")
set(CPACK_RPM_PACKAGE_URL "https://github.com/STEllAR-GROUP/hpx")
set(CPACK_RPM_PACKAGE_SUMMARY
    "The C++ Standard Library for Parallelism and Concurrency (HPX)"
)
set(CPACK_RPM_PACKAGE_DESCRIPTION
    "The C++ Standard Library for Parallelism and Concurrency compiled for educational usage"
)
set(CPACK_RPM_PACKAGE_REQUIRES "boost, hwloc, gperftools")
set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}"
)
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    /etc/ld.so.conf.d
    /usr/local
    /usr/local/bin
    /usr/local/.build-id
    /usr/local/include
    /usr/local/share
    /usr/local/lib
)
set(CPACK_RPM_CHANGELOG_FILE
    "${CMAKE_HOME_DIRECTORY}/cmake/packaging/rpm/Changelog.txt"
)
include(CPack)
