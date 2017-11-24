#!/bin/bash -x
#  Copyright (c) 2017 Agustin Berge
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

make -j2 -k tools.inspect
./bin/inspect --all --output=./hpx_inspect_report.html /hpx
RESULT=$?

mkdir -p ${1}/
cp ./hpx_inspect_report.html ${1}/
exit ${RESULT}
