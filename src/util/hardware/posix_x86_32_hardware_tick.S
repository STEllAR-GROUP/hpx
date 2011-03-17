################################################################################
#   Copyright (c) 2011 Bryce Lelbach
#
#   Distributed under the Boost Software License, Version 1.0. (See accompanying
#   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
################################################################################

# This is an assembly source file. Please note:
#   0.) This code is written for the x86-32 ISA.
#   1.) This code uses the AT&T x86 assembly language syntax.
#   2.) This source file must be preprocessed with a standard compliant C
#       preprocessor (clang-linux, intel-linux and gcc support this).
#   3.) This source file targets clang-linux, intel-linux and gcc.

.file "posix_x86_32_hardware_tick.S"
  
.text
.p2align 4,,15
.globl HPX_posix_x86_32_hardware_tick
.type HPX_posix_x86_32_hardware_tick, @function

HPX_posix_x86_32_hardware_tick:
  #############################################################################
  # Setup call frame information (CFI) stuff.
  .cfi_startproc # Initialize this functions entry in .eh_frame.

  ############################################################################
  # Perform the time stamp read.

  # First, we need to call a serializing instruction.
  #if defined(HPX_USE_SSE2)
    mfence # AMD says mfence, intel says lfence, mfence makes more sense to me.
  #else
    cpuid # cpuid is not ideal because it's expensive and uses 4 registers.
  #endif
  rdtsc # Read the the timestamp counter.

  #############################################################################
  # Return.
  ret           # Return to the caller.
  .cfi_endproc  # Clean up CFI jazz.

