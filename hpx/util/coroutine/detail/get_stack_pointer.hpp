//  Copyright (c) 2013-2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COROUTINE_GET_STACK_POINTER_HPP
#define HPX_COROUTINE_GET_STACK_POINTER_HPP

#if !defined(HPX_WINDOWS)
#if defined(__x86_64__) || defined(__amd64)                                    \
    || defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) \
    || defined(__powerpc__)                                                    \
    || defined(__arm__)
#define HPX_HAVE_THREADS_GET_STACK_POINTER
#endif

#include <boost/cstdint.hpp>

#include <limits>

namespace hpx { namespace util { namespace coroutines { namespace detail
{
    inline boost::uintptr_t get_stack_ptr()
    {
        boost::uintptr_t stack_ptr = (std::numeric_limits<boost::uintptr_t>::max)();
#if defined(__x86_64__) || defined(__amd64)
        asm("movq %%rsp, %0" : "=r"(stack_ptr));
#elif defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__)
        asm("movl %%esp, %0" : "=r"(stack_ptr));
#elif defined(__powerpc__)
        std::size_t stack_ptr_p = &stack_ptr;
        asm("stw %%r1, 0(%0)" : "=&r"(stack_ptr_p));
#elif defined(__arm__)
        asm("mov %0, sp" : "=r"(stack_ptr));
#endif
        return stack_ptr;
    }
}}}}

#endif
#endif
