#/usr/bin/env bash
#
# Copyright (c) 2019 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file BOOST_LICENSE_1_0.rst or copy at http://www.boost.org/LICENSE_1_0.txt)

if [ "$TRAVIS_OS_NAME" = "windows" ]; then
    cmake \
        -H. \
        -Bbuild \
        -G'Visual Studio 15 2017 Win64' \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_TOOLCHAIN_FILE='C:/projects/vcpkg/scripts/buildsystems/vcpkg.cmake' \
        -DHPX_WITH_PSEUDO_DEPENDENCIES=ON \
        -DHPX_WITH_EXAMPLES=OFF \
        -DHPX_WITH_TESTS=OFF \
        -DHPX_WITH_DEPRECATION_WARNINGS=OFF
elif [ "$TRAVIS_OS_NAME" = "osx" ]; then
    cmake \
        -H. \
        -Bbuild \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DHPX_WITH_EXAMPLES=ON \
        -DHPX_WITH_TESTS=ON
else
    echo "no scripts set up for \"$TRAVIS_OS_NAME\""
    exit 1
fi
