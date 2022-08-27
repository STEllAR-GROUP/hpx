//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2007 Robert Perricone
//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2013-2016 Thomas Heller
//  Copyright (c) 2017 Christopher Taylor
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(__linux) || defined(linux) || defined(__linux__)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/coroutines/detail/posix_utility.hpp>
#include <hpx/coroutines/detail/swap_context.hpp>
#include <hpx/coroutines/signal_handler_debugging.hpp>
#include <hpx/debugging/attach_debugger.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/util/from_string.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/version.hpp>

#if defined(HPX_HAVE_STACKTRACES)
#include <hpx/debugging/backtrace.hpp>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <sys/param.h>

#if defined(HPX_HAVE_STACKOVERFLOW_DETECTION)

#include <cstring>
#include <signal.h>
#include <stdlib.h>
#include <strings.h>

#if !defined(SEGV_STACK_SIZE)
#define SEGV_STACK_SIZE MINSIGSTKSZ + 4096
#endif

#endif

#include <iomanip>
#include <iostream>

#if defined(HPX_HAVE_VALGRIND)
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wpointer-arith"
#endif
#include <valgrind/valgrind.h>
#endif

#if defined(HPX_HAVE_ADDRESS_SANITIZER)
#include <sanitizer/asan_interface.h>
#endif

// Defining HPX_COROUTINE_NO_SEPARATE_CALL_SITES will disable separate
// invoke, and yield swap_context functions. Separate calls sites
// increase performance by 25% at least on P4 for invoke+yield back loops
// at the cost of a slightly higher instruction cache use and is thus enabled by
// default.

#if defined(__x86_64__)
extern "C" void swapcontext_stack(void***, void**) noexcept;
extern "C" void swapcontext_stack2(void***, void**) noexcept;
#else
extern "C" void swapcontext_stack(void***, void**) noexcept
    __attribute((regparm(2)));
extern "C" void swapcontext_stack2(void***, void**) noexcept
    __attribute((regparm(2)));
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx::threads::coroutines {

#if defined(HPX_HAVE_STACKOVERFLOW_DETECTION) &&                               \
    !defined(HPX_HAVE_ADDRESS_SANITIZER)

// heuristic value 1 kilobyte
#define COROUTINE_STACKOVERFLOW_ADDR_EPSILON 1000UL

    static inline void check_coroutine_stack_overflow(
        siginfo_t* infoptr, void* ctxptr)
    {
        ucontext_t* uc_ctx = static_cast<ucontext_t*>(ctxptr);
        char* sigsegv_ptr = static_cast<char*>(infoptr->si_addr);

        // https://www.gnu.org/software/libc/manual/html_node/Signal-Stack.html
        //
        char* stk_ptr = static_cast<char*>(uc_ctx->uc_stack.ss_sp);

        std::ptrdiff_t addr_delta = (sigsegv_ptr > stk_ptr) ?
            (sigsegv_ptr - stk_ptr) :
            (stk_ptr - sigsegv_ptr);

        // check the stack addresses, if they're < 10 apart, terminate
        // program should filter segmentation faults caused by
        // coroutine stack overflows from 'genuine' stack overflows
        //
        if (static_cast<size_t>(addr_delta) <
            COROUTINE_STACKOVERFLOW_ADDR_EPSILON)
        {
            std::cerr << "Stack overflow in coroutine at address "
                      << std::internal << std::hex
                      << std::setw(sizeof(sigsegv_ptr) * 2 + 2)
                      << std::setfill('0') << sigsegv_ptr << ".\n\n";

            std::cerr
                << "Configure the hpx runtime to allocate a larger coroutine "
                   "stack size.\n Use the hpx.stacks.small_size, "
                   "hpx.stacks.medium_size,\n hpx.stacks.large_size, or "
                   "hpx.stacks.huge_size configuration\nflags to configure "
                   "coroutine stack sizes.\n"
                << std::endl;
        }
    }

    static inline void sigsegv_handler(
        int signum, siginfo_t* infoptr, void* ctxptr)
    {
        if (attach_debugger_on_sigv)
        {
            hpx::util::attach_debugger();
        }

        if (diagnostics_on_terminate)
        {
            int const verbosity = exception_verbosity;

            if (verbosity >= 2)
            {
                std::cerr << full_build_string() << "\n";
            }

#if defined(HPX_HAVE_STACKTRACES)
            if (verbosity >= 1)
            {
                std::cerr << "{stack-trace}: " << hpx::util::trace(trace_depth)
                          << "\n";
            }
#endif

            char* reason = strsignal(signum);
            std::cerr << "{what}: " << (reason ? reason : "Unknown signal")
                      << std::endl;

            check_coroutine_stack_overflow(infoptr, ctxptr);

            std::abort();
        }
    }
#endif

    // some platforms need special preparation of the main thread
    struct prepare_main_thread
    {
        constexpr prepare_main_thread() = default;
    };
}    // namespace hpx::threads::coroutines

namespace hpx::threads::coroutines::detail::lx {

    template <typename T>
    HPX_FORCEINLINE void trampoline(void* fun)
    {
        (*static_cast<T*>(fun))();
        std::abort();
    }

    template <typename CoroutineImpl>
    class x86_linux_context_impl;

    class x86_linux_context_impl_base : detail::context_impl_base
    {
    public:
        x86_linux_context_impl_base()
          : m_sp(nullptr)
#if defined(HPX_HAVE_ADDRESS_SANITIZER)
          , asan_fake_stack(nullptr)
          , asan_stack_bottom(nullptr)
          , asan_stack_size(0)
#endif
        {
        }

        void prefetch() const
        {
#if defined(__x86_64__)
            HPX_ASSERT(sizeof(void*) == 8);
#else
            HPX_ASSERT(sizeof(void*) == 4);
#endif

            __builtin_prefetch(m_sp, 1, 3);
            __builtin_prefetch(m_sp, 0, 3);
            __builtin_prefetch(
                static_cast<void**>(m_sp) + 64 / sizeof(void*), 1, 3);
            __builtin_prefetch(
                static_cast<void**>(m_sp) + 64 / sizeof(void*), 0, 3);
#if !defined(__x86_64__)
            __builtin_prefetch(
                static_cast<void**>(m_sp) + 32 / sizeof(void*), 1, 3);
            __builtin_prefetch(
                static_cast<void**>(m_sp) + 32 / sizeof(void*), 0, 3);
            __builtin_prefetch(
                static_cast<void**>(m_sp) - 32 / sizeof(void*), 1, 3);
            __builtin_prefetch(
                static_cast<void**>(m_sp) - 32 / sizeof(void*), 0, 3);
#endif
            __builtin_prefetch(
                static_cast<void**>(m_sp) - 64 / sizeof(void*), 1, 3);
            __builtin_prefetch(
                static_cast<void**>(m_sp) - 64 / sizeof(void*), 0, 3);
        }

        // Free function. Saves the current context in @p from
        // and restores the context in @p to.
        // @note This function is found by ADL.
        friend void swap_context(x86_linux_context_impl_base& from,
            x86_linux_context_impl_base const& to, default_hint);

        friend void swap_context(x86_linux_context_impl_base& from,
            x86_linux_context_impl_base const& to, yield_hint);

#if defined(HPX_HAVE_ADDRESS_SANITIZER)
        void start_switch_fiber(void** fake_stack)
        {
            __sanitizer_start_switch_fiber(
                fake_stack, asan_stack_bottom, asan_stack_size);
        }
        void start_yield_fiber(
            void** fake_stack, x86_linux_context_impl_base& caller)
        {
            __sanitizer_start_switch_fiber(
                fake_stack, caller.asan_stack_bottom, caller.asan_stack_size);
        }
        void finish_yield_fiber(void* fake_stack)
        {
            __sanitizer_finish_switch_fiber(
                fake_stack, &asan_stack_bottom, &asan_stack_size);
        }
        void finish_switch_fiber(
            void* fake_stack, x86_linux_context_impl_base& caller)
        {
            __sanitizer_finish_switch_fiber(
                fake_stack, &caller.asan_stack_bottom, &caller.asan_stack_size);
        }
#endif

    protected:
        void** m_sp;

#if defined(HPX_HAVE_ADDRESS_SANITIZER)
    public:
        void* asan_fake_stack;
        void const* asan_stack_bottom;
        std::size_t asan_stack_size;
#endif
    };

    template <typename CoroutineImpl>
    class x86_linux_context_impl : public x86_linux_context_impl_base
    {
    public:
        enum
        {
            default_stack_size = 4 * EXEC_PAGESIZE
        };

        typedef x86_linux_context_impl_base context_impl_base;

        // Create a context that on restore invokes Functor on
        //  a new stack. The stack size can be optionally specified.
        explicit x86_linux_context_impl(std::ptrdiff_t stack_size = -1)
          : m_stack_size(stack_size == -1 ?
                    static_cast<std::ptrdiff_t>(default_stack_size) :
                    stack_size)
          , m_stack(nullptr)
        {
        }

        void init()
        {
            if (m_stack != nullptr)
                return;

            if (0 != (m_stack_size % EXEC_PAGESIZE))
            {
                throw std::runtime_error(
                    hpx::util::format("stack size of {1} is not page "
                                      "aligned, page size is {2}",
                        m_stack_size, EXEC_PAGESIZE));
            }

            if (0 >= m_stack_size)
            {
                throw std::runtime_error(hpx::util::format(
                    "stack size of {1} is invalid", m_stack_size));
            }

            m_stack =
                posix::alloc_stack(static_cast<std::size_t>(m_stack_size));
            if (m_stack == nullptr)
            {
                throw std::runtime_error("could not allocate memory for stack");
            }

            posix::watermark_stack(
                m_stack, static_cast<std::size_t>(m_stack_size));

            typedef void fun(void*);
            fun* funp = trampoline<CoroutineImpl>;

            m_sp = (static_cast<void**>(m_stack) +
                       static_cast<std::size_t>(m_stack_size) / sizeof(void*)) -
                context_size;

            m_sp[cb_idx] = this;
            m_sp[funp_idx] = reinterpret_cast<void*>(funp);

#if defined(HPX_HAVE_VALGRIND) && !defined(NVALGRIND)
            {
                void* eos = static_cast<char*>(m_stack) + m_stack_size;
                m_sp[valgrind_id_idx] = reinterpret_cast<void*>(
                    VALGRIND_STACK_REGISTER(m_stack, eos));
            }
#endif
#if defined(HPX_HAVE_ADDRESS_SANITIZER)
            asan_stack_size = m_stack_size;
            asan_stack_bottom = const_cast<const void*>(m_stack);
#endif

            set_sigsegv_handler();
        }

        ~x86_linux_context_impl()
        {
            if (m_stack)
            {
#if defined(HPX_HAVE_VALGRIND) && !defined(NVALGRIND)
                VALGRIND_STACK_DEREGISTER(
                    reinterpret_cast<std::size_t>(m_sp[valgrind_id_idx]));
#endif
                posix::free_stack(
                    m_stack, static_cast<std::size_t>(m_stack_size));
            }
        }

        // Return the size of the reserved stack address space.
        std::ptrdiff_t get_stacksize() const
        {
            return m_stack_size;
        }

        void reset_stack()
        {
            HPX_ASSERT(m_stack);
            if (posix::reset_stack(
                    m_stack, static_cast<std::size_t>(m_stack_size)))
            {
#if defined(HPX_HAVE_COROUTINE_COUNTERS)
                increment_stack_unbind_count();
#endif
            }
        }

        void rebind_stack()
        {
            HPX_ASSERT(m_stack);
#if defined(HPX_HAVE_COROUTINE_COUNTERS)
            increment_stack_recycle_count();
#endif

            // On rebind, we initialize our stack to ensure a virgin stack
            m_sp = (static_cast<void**>(m_stack) +
                       static_cast<std::size_t>(m_stack_size) / sizeof(void*)) -
                context_size;

            typedef void fun(void*);
            fun* funp = trampoline<CoroutineImpl>;
            m_sp[cb_idx] = this;
            m_sp[funp_idx] = reinterpret_cast<void*>(funp);
#if defined(HPX_HAVE_ADDRESS_SANITIZER)
            asan_stack_size = m_stack_size;
            asan_stack_bottom = const_cast<const void*>(m_stack);
#endif
        }

        std::ptrdiff_t get_available_stack_space()
        {
            return get_stack_ptr() - reinterpret_cast<std::size_t>(m_stack) -
                context_size;
        }

        using counter_type = std::atomic<std::int64_t>;

#if defined(HPX_HAVE_COROUTINE_COUNTERS)
    private:
        static counter_type& get_stack_unbind_counter()
        {
            static counter_type counter(0);
            return counter;
        }

        static counter_type& get_stack_recycle_counter()
        {
            static counter_type counter(0);
            return counter;
        }

        static std::uint64_t increment_stack_unbind_count()
        {
            return ++get_stack_unbind_counter();
        }

        static std::uint64_t increment_stack_recycle_count()
        {
            return ++get_stack_recycle_counter();
        }

    public:
        static std::uint64_t get_stack_unbind_count(bool reset)
        {
            return util::get_and_reset_value(get_stack_unbind_counter(), reset);
        }

        static std::uint64_t get_stack_recycle_count(bool reset)
        {
            return util::get_and_reset_value(
                get_stack_recycle_counter(), reset);
        }
#endif

        friend void swap_context(x86_linux_context_impl_base& from,
            x86_linux_context_impl_base const& to, default_hint);

        friend void swap_context(x86_linux_context_impl_base& from,
            x86_linux_context_impl_base const& to, yield_hint);

    private:
        void set_sigsegv_handler()
        {
#if defined(HPX_HAVE_STACKOVERFLOW_DETECTION) &&                               \
    !defined(HPX_HAVE_ADDRESS_SANITIZER)
            // concept inspired by the following links:
            //
            // https://rethinkdb.com/blog/handling-stack-overflow-on-custom-stacks/
            // http://www.evanjones.ca/software/threading.html
            //
            segv_stack.ss_sp = valloc(SEGV_STACK_SIZE);
            segv_stack.ss_flags = 0;
            segv_stack.ss_size = SEGV_STACK_SIZE;

            std::memset(&action, '\0', sizeof(action));
            action.sa_flags = SA_SIGINFO | SA_ONSTACK;
            action.sa_sigaction = &sigsegv_handler;

            sigaltstack(&segv_stack, nullptr);
            sigemptyset(&action.sa_mask);
            sigaddset(&action.sa_mask, SIGSEGV);
            sigaction(SIGSEGV, &action, nullptr);
#endif
        }

#if defined(__x86_64__)
        // structure of context_data:
        // 11: additional alignment (or valgrind_id if enabled)
        // 10: parm 0 of trampoline
        // 9:  dummy return address for trampoline
        // 8:  return addr (here: start addr)
        // 7:  rbp
        // 6:  rbx
        // 5:  rsi
        // 4:  rdi
        // 3:  r12
        // 2:  r13
        // 1:  r14
        // 0:  r15
#if defined(HPX_HAVE_VALGRIND) && !defined(NVALGRIND)
        static const std::size_t valgrind_id_idx = 11;
#endif

        static const std::size_t context_size = 12;
        static const std::size_t cb_idx = 10;
        static const std::size_t funp_idx = 8;
#else
        // structure of context_data:
        // 7: valgrind_id (if enabled)
        // 6: parm 0 of trampoline
        // 5: dummy return address for trampoline
        // 4: return addr (here: start addr)
        // 3: ebp
        // 2: ebx
        // 1: esi
        // 0: edi
#if defined(HPX_HAVE_VALGRIND) && !defined(NVALGRIND)
        static const std::size_t valgrind_id_idx = 7;
#endif
        static const std::size_t context_size = 8;
        static const std::size_t cb_idx = 6;
        static const std::size_t funp_idx = 4;
#endif

        std::ptrdiff_t m_stack_size;
        void* m_stack;

#if defined(HPX_HAVE_STACKOVERFLOW_DETECTION) &&                               \
    !defined(HPX_HAVE_ADDRESS_SANITIZER)
        struct sigaction action;
        stack_t segv_stack;
#endif
    };

    // Free function. Saves the current context in @p from
    // and restores the context in @p to.
    // @note This function is found by ADL.
    inline void swap_context(x86_linux_context_impl_base& from,
        x86_linux_context_impl_base const& to, default_hint)
    {
        // HPX_ASSERT(*(void**)to.m_stack == (void*)~0);
        to.prefetch();
        swapcontext_stack(&from.m_sp, to.m_sp);
    }

    inline void swap_context(x86_linux_context_impl_base& from,
        x86_linux_context_impl_base const& to, yield_hint)
    {
        // HPX_ASSERT(*(void**)from.m_stack == (void*)~0);
        to.prefetch();
#if !defined(HPX_COROUTINE_NO_SEPARATE_CALL_SITES)
        swapcontext_stack2(&from.m_sp, to.m_sp);
#else
        swapcontext_stack(&from.m_sp, to.m_sp);
#endif
    }
}    // namespace hpx::threads::coroutines::detail::lx

#if defined(HPX_HAVE_VALGRIND)
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif
#endif

#else

#error This header can only be included when compiling for linux systems.

#endif
