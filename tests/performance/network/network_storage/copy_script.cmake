# Copyright (c) 2014 John Biddiscombe
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

FILE(TO_NATIVE_PATH "${SCRIPT_SOURCE_DIR}/${SCRIPT_NAME}.sh.in" INFILE) 
FILE(TO_NATIVE_PATH "${SCRIPT_DEST_DIR}/${SCRIPT_NAME}.sh" OUTFILE) 

STRING(REPLACE "\"" "" FILE1 "${INFILE}")
STRING(REPLACE "\"" "" FILE2 "${OUTFILE}")

configure_file(
  "${FILE1}"
  "${FILE2}"
  @ONLY
)
