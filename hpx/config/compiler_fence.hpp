//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_COMPILER_FENCE_HPP
#define HPX_CONFIG_COMPILER_FENCE_HPP

#include <hpx/config/compiler_specific.hpp>

#if defined(__INTEL_COMPILER)

#define HPX_COMPILER_FENCE __memory_barrier();

#elif defined( _MSC_VER ) && _MSC_VER >= 1310

extern "C" void _ReadWriteBarrier();
#pragma intrinsic( _ReadWriteBarrier )

#define HPX_COMPILER_FENCE _ReadWriteBarrier();

#elif defined(__GNUC__)

#define HPX_COMPILER_FENCE __asm__ __volatile__( "" : : : "memory" );

#else

#define HPX_COMPILER_FENCE

#endif

#endif
