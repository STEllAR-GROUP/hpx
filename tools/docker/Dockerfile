# Copyright (c) 2015 Martin Stumpf
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

FROM stellargroup/build_env:debian_clang

ADD . /hpx

RUN cd /hpx/build && make install && cd /
RUN rm -rf /hpx

RUN ldconfig

CMD bash
WORKDIR /
