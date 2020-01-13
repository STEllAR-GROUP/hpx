#/usr/bin/env bash
#
# Copyright (c) 2019 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file BOOST_LICENSE_1_0.rst or copy at http://www.boost.org/LICENSE_1_0.txt)

if [ "$TRAVIS_OS_NAME" = "windows" ]; then
    cmake \
        --build build \
        --config Debug \
        --target core \
        -- -verbosity:minimal -maxcpucount:4 -nologo
elif [ "$TRAVIS_OS_NAME" = "osx" ]; then
    cmake --build build --target core || exit 1
    cmake --build build --target examples || exit 1
    ctest --output-on-failure -R tests.examples
else
    echo "no scripts set up for \"$TRAVIS_OS_NAME\""
    exit 1
fi
