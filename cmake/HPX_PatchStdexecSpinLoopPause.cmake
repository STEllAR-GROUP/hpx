#  Copyright (c) 2026 Hartmut Kaiser
#
#  SPDX-License-Identifier: BSL-1.0
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This patch file works around a problem seen on ARM gcc 11.4.0, see:
# https://github.com/NVIDIA/stdexec/issues/2150

if(NOT DEFINED STDEXEC_SPIN_LOOP_PAUSE_FILE)
  message(FATAL_ERROR "STDEXEC_SPIN_LOOP_PAUSE_FILE must be defined")
endif()

if(NOT EXISTS "${STDEXEC_SPIN_LOOP_PAUSE_FILE}")
  message(FATAL_ERROR "File not found: ${STDEXEC_SPIN_LOOP_PAUSE_FILE}")
endif()

file(READ "${STDEXEC_SPIN_LOOP_PAUSE_FILE}" _stdexec_spin_loop_pause_contents)

string(
  REPLACE "STDEXEC_ATTRIBUTE(always_inline) inline"
          "STDEXEC_ATTRIBUTE(always_inline)" _stdexec_spin_loop_pause_contents
          "${_stdexec_spin_loop_pause_contents}"
)

file(WRITE "${STDEXEC_SPIN_LOOP_PAUSE_FILE}"
     "${_stdexec_spin_loop_pause_contents}"
)
