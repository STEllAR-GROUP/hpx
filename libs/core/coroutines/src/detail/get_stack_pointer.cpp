//  Copyright (c) 2013-2016 Thomas Heller
//  Copyright (c) 2022 Christopher Taylor
//  Copyright (c) 2024 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/coroutines/detail/get_stack_pointer.hpp>

#if !defined(HPX_WINDOWS)

#include <cstddef>
#include <limits>

namespace hpx::threads::coroutines::detail {

    std::size_t get_stack_ptr() noexcept
    {
#if defined(HPX_HAVE_BUILTIN_FRAME_ADDRESS)
        return std::size_t(__builtin_frame_address(0));
#else
        std::size_t stack_ptr = (std::numeric_limits<std::size_t>::max)();
#if defined(__x86_64__) || defined(__amd64)
        asm("movq %%rsp, %0" : "=r"(stack_ptr));
#elif defined(__i386__) || defined(__i486__) || defined(__i586__) ||           \
    defined(__i686__)
        asm("movl %%esp, %0" : "=r"(stack_ptr));
#elif defined(__powerpc__)
        void* stack_ptr_p = &stack_ptr;
        asm("stw %%r1, 0(%0)" : "=&r"(stack_ptr_p));
#elif defined(__arm__)
        asm("mov %0, sp" : "=r"(stack_ptr));
#elif defined(__riscv)
        __asm__ __volatile__("add %0, x0, sp" : "=r"(stack_ptr));
#endif
        return stack_ptr;
#endif
    }
}    // namespace hpx::threads::coroutines::detail
#endif
