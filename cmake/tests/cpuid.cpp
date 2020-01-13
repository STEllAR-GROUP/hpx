////////////////////////////////////////////////////////////////////////////////
//  Copyright 2003 & onward LASMEA UMR 6602 CNRS/Univ. Clermont II
//  Copyright 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <string>
using namespace std;

struct registers_t { uint32_t eax, ebx, ecx, edx; };

#if defined __GNUC__
void __cpuid(registers_t& CPUInfo, uint32_t InfoType)
{
  __asm__ __volatile__
    (
     "cpuid":
       "=a" (CPUInfo.eax), "=b" (CPUInfo.ebx)
     , "=c" (CPUInfo.ecx), "=d" (CPUInfo.edx)
     : "a" (InfoType)
     );
}

#elif defined _MSC_VER
#include <intrin.h>
#endif

bool has_bit_set(uint32_t value, uint32_t bit)
{
  return (value & (1U<<bit)) != 0;
}

struct matcher
{
  uint32_t function;
  uint32_t registers_t::* reg;
  uint32_t bit;
  const char* target;
}
options[] =
  {
    {0x00000001U, &registers_t::edx, 19, "clflush"}
  , {0x00000001U, &registers_t::edx,  8, "cx8"    }
  , {0x00000001U, &registers_t::ecx, 13, "cx16"   }
  , {0x00000001U, &registers_t::edx, 15, "cmov"   }
  , {0x00000001U, &registers_t::edx,  5, "msr"    }
  , {0x00000001U, &registers_t::edx,  4, "rdtsc"  }
  , {0x80000001U, &registers_t::edx, 27, "rdtscp" }
  , {0x00000001U, &registers_t::edx, 23, "mmx"    }
  , {0x00000001U, &registers_t::edx, 25, "sse"    }
  , {0x00000001U, &registers_t::edx, 26, "sse2"   }
  , {0x00000001U, &registers_t::ecx,  0, "sse3"   }
  , {0x00000001U, &registers_t::ecx,  9, "ssse3"  }
  , {0x00000001U, &registers_t::ecx, 19, "sse4.1" }
  , {0x00000001U, &registers_t::ecx, 20, "sse4.2" }
  , {0x00000001U, &registers_t::ecx, 28, "avx"    }
  , {0x80000001U, &registers_t::edx, 11, "xop"    }
  , {0x80000001U, &registers_t::edx, 16, "fma4"   }
};
const size_t noptions = sizeof options / sizeof options[0];

int main(int argc, char** argv)
{
  registers_t registers;
  if (argc < 2) return -1;

  string target(argv[1]);
  __cpuid(registers,0x00000000);

  matcher m;
  size_t i = 0;
  for (i=0; i<noptions; ++i) {
    if (target == options[i].target) {
      m = options[i];
      break;
    }
  }
  if (i >= noptions) return -2;

  __cpuid(registers, m.function);

  // exit with 0 if the bit is set
  return !has_bit_set(registers.*m.reg, m.bit);
}
