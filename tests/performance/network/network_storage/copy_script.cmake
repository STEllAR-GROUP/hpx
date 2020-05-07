# Copyright (c) 2014 John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

file(TO_NATIVE_PATH "${SCRIPT_SOURCE_DIR}/${SCRIPT_NAME}.sh.in" INFILE)
file(TO_NATIVE_PATH "${SCRIPT_DEST_DIR}/${SCRIPT_NAME}.sh" OUTFILE)

string(REPLACE "\"" "" FILE1 "${INFILE}")
string(REPLACE "\"" "" FILE2 "${OUTFILE}")

configure_file("${FILE1}" "${FILE2}" @ONLY)
