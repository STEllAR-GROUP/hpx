#/usr/bin/env bash
#
# Copyright (c) 2019 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file BOOST_LICENSE_1_0.rst or copy at http://www.boost.org/LICENSE_1_0.txt)

if [ "$TRAVIS_OS_NAME" = "windows" ]; then
    mkdir 'C:/projects'
    wget \
        --output-document='C:/projects/vcpkg-export-hpx-dependencies.7z' \
        'http://stellar-group.org/files/vcpkg-export-hpx-dependencies.7z'
    7z x \
        'C:/projects/vcpkg-export-hpx-dependencies.7z' \
        -y \
        -o'C:/projects/vcpkg' \
        >NUL
elif [ "$TRAVIS_OS_NAME" = "osx" ]; then
    # We do not upgrade Boost because it takes a long time
    brew update && \
        brew install hwloc gperftools ninja &&
        brew upgrade cmake
else
    echo "no scripts set up for \"$TRAVIS_OS_NAME\""
    exit 1
fi
