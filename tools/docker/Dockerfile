# Copyright (c) 2018 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

FROM stellargroup/build_env:ubuntu

# docker build needs to be run in /usr/local
ADD . /usr/local
RUN ldconfig
WORKDIR /
CMD bash
