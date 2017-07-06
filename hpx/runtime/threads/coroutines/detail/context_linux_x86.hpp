//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2007 Robert Perricone
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2013-2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_COROUTINES_DETAIL_CONTEXT_LINUX_HPP
#define HPX_RUNTIME_THREADS_COROUTINES_DETAIL_CONTEXT_LINUX_HPP

#if defined(__linux) || defined(linux) || defined(__linux__)

#include <hpx/config.hpp>
#include <hpx/runtime/threads/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/runtime/threads/coroutines/detail/posix_utility.hpp>
#include <hpx/runtime/threads/coroutines/detail/swap_context.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/get_and_reset_value.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <sys/param.h>

#include <boost/atomic.hpp>
#include <boost/format.hpp>

#include <signal.h>
#include <stdlib.h>
#include <strings.h>

#ifndef SEGV_STACK_SIZE
  #define SEGV_STACK_SIZE MINSIGSTKSZ+4096
#endif

#include <iostream>
#include <iomanip>

#if defined(HPX_HAVE_VALGRIND)
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wpointer-arith"
#endif
#include <valgrind/valgrind.h>
#endif

/*
 * Defining HPX_COROUTINE_NO_SEPARATE_CALL_SITES will disable separate
 * invoke, and yield swap_context functions. Separate calls sites
 * increase performance by 25% at least on P4 for invoke+yield back loops
 * at the cost of a slightly higher instruction cache use and is thus enabled by
 * default.
 */

#if defined(__x86_64__)
extern "C" void swapcontext_stack (void***, void**) throw();
extern "C" void swapcontext_stack2 (void***, void**) throw();
#else
extern "C" void swapcontext_stack (void***, void**) throw() __attribute((regparm(2)));
extern "C" void swapcontext_stack2 (void***, void**) throw()__attribute((regparm(2)));
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace coroutines
{
    // some platforms need special preparation of the main thread
    struct prepare_main_thread
    {
        prepare_main_thread() {}
        ~prepare_main_thread() {}
    };

    namespace detail { namespace lx
    {
        template <typename TO, typename FROM>
        TO nasty_cast(FROM f)
        {
            union {
                FROM f; TO t;
            } u;
            u.f = f;
            return u.t;
        }

        template<typename T>
        HPX_FORCEINLINE void trampoline(T* fun);

        template<typename T>
        void trampoline(T* fun)
        {
            (*fun)();
            std::abort();
        }

        class x86_linux_context_impl;

        class x86_linux_context_impl_base : detail::context_impl_base
        {
        public:
            x86_linux_context_impl_base() {}

            void prefetch() const
            {
#if defined(__x86_64__)
                HPX_ASSERT(sizeof(void*) == 8);
#else
                HPX_ASSERT(sizeof(void*) == 4);
#endif

                __builtin_prefetch(m_sp, 1, 3);
                __builtin_prefetch(m_sp, 0, 3);
                __builtin_prefetch(static_cast<void**>(m_sp) + 64 / sizeof(void*), 1, 3);
                __builtin_prefetch(static_cast<void**>(m_sp) + 64 / sizeof(void*), 0, 3);
#if !defined(__x86_64__)
                __builtin_prefetch(static_cast<void**>(m_sp) + 32 / sizeof(void*), 1, 3);
                __builtin_prefetch(static_cast<void**>(m_sp) + 32 / sizeof(void*), 0, 3);
                __builtin_prefetch(static_cast<void**>(m_sp) - 32 / sizeof(void*), 1, 3);
                __builtin_prefetch(static_cast<void**>(m_sp) - 32 / sizeof(void*), 0, 3);
#endif
                __builtin_prefetch(static_cast<void**>(m_sp) - 64 / sizeof(void*), 1, 3);
                __builtin_prefetch(static_cast<void**>(m_sp) - 64 / sizeof(void*), 0, 3);
            }

            /**
             * Free function. Saves the current context in @p from
             * and restores the context in @p to.
             * @note This function is found by ADL.
             */
            friend void swap_context(x86_linux_context_impl_base& from,
                x86_linux_context_impl_base const& to, default_hint);

            friend void swap_context(x86_linux_context_impl_base& from,
                x86_linux_context_impl_base const& to, yield_hint);

        protected:
            void ** m_sp;
        };

        class x86_linux_context_impl : public x86_linux_context_impl_base
        {
        public:
            enum { default_stack_size = 4 * EXEC_PAGESIZE };

            typedef x86_linux_context_impl_base context_impl_base;

            x86_linux_context_impl()
                : m_stack(nullptr)
#if defined(HPX_HAVE_THREAD_STACKOVERFLOW_DETECTION)
            {
                segv_stack.ss_sp = valloc(SEGV_STACK_SIZE);
                segv_stack.ss_flags = 0;
                segv_stack.ss_size = SEGV_STACK_SIZE;

                bzero(&action, sizeof(action));
                action.sa_flags = SA_SIGINFO|SA_ONSTACK; //SA_STACK
                action.sa_sigaction = &sigsegv_handler;

                sigaltstack(&segv_stack, nullptr);
                sigfillset(&action.sa_mask);
                sigaction(SIGSEGV, &action, nullptr);
            }
#else
            {}
#endif

            /**
             * Create a context that on restore invokes Functor on
             *  a new stack. The stack size can be optionally specified.
             */
            template<typename Functor>
            x86_linux_context_impl(Functor& cb, std::ptrdiff_t stack_size = -1)
              : m_stack_size(stack_size == -1
                  ? static_cast<std::ptrdiff_t>(default_stack_size)
                  : stack_size),
                m_stack(nullptr)
            {
                if (0 != (m_stack_size % EXEC_PAGESIZE))
                {
                    throw std::runtime_error(
                        boost::str(boost::format(
                            "stack size of %1% is not page aligned, page size is %2%")
                            % m_stack_size % EXEC_PAGESIZE));
                }

                if (0 >= m_stack_size)
                {
                    throw std::runtime_error(
                        boost::str(boost::format("stack size of %1% is invalid") %
                            m_stack_size));
                }

                m_stack = posix::alloc_stack(static_cast<std::size_t>(m_stack_size));
                HPX_ASSERT(m_stack);
                posix::watermark_stack(m_stack, static_cast<std::size_t>(m_stack_size));

                typedef void fun(Functor*);
                fun * funp = trampoline;

                m_sp = (static_cast<void**>(m_stack)
                    + static_cast<std::size_t>(m_stack_size) / sizeof(void*))
                    - context_size;

                m_sp[backup_cb_idx] = m_sp[cb_idx] = &cb;
                m_sp[backup_funp_idx] = m_sp[funp_idx] = nasty_cast<void*>(funp);

#if defined(HPX_HAVE_VALGRIND) && !defined(NVALGRIND)
                {
                    void * eos = static_cast<char*>(m_stack) + m_stack_size;
                    m_sp[valgrind_id_idx] = reinterpret_cast<void*>(
                        VALGRIND_STACK_REGISTER(m_stack, eos));
                }
#endif

#if defined(HPX_HAVE_THREAD_STACKOVERFLOW_DETECTION)
                segv_stack.ss_sp = valloc(SEGV_STACK_SIZE);
                segv_stack.ss_flags = 0;
                segv_stack.ss_size = SEGV_STACK_SIZE;

                bzero(&action, sizeof(action));
                action.sa_flags = SA_SIGINFO|SA_ONSTACK; //SA_STACK
                action.sa_sigaction = &sigsegv_handler;

                sigaltstack(&segv_stack, NULL);
                sigfillset(&action.sa_mask);
                sigaction(SIGSEGV, &action, NULL);
#endif
            }

#if defined(HPX_HAVE_THREAD_STACKOVERFLOW_DETECTION)
            static void sigsegv_handler(int signum, siginfo_t *info,
                void *data)
            {
                void *addr = info->si_addr;

                std::cerr << "Stack overflow in coroutine at address "
                    << std::internal << std::hex
                    << std::setw(sizeof(addr)*2+2)
                    << std::setfill('0') << static_cast<int*>(addr)
                    << "." << std::endl;

                std::cerr << std::endl << "Configure the hpx runtime to "
                    << "allocate a larger "
                    << "coroutine stack size." << std::endl
                    << "Use the hpx.stacks.small_size, "
                    << "hpx.stacks.medium_size, "
                    << std::endl
                    << "hpx.stacks.large_size, "
                    << "or hpx.stacks.huge_size runtime "
                    << std::endl
                    << "flags to configure coroutine heap sizes." 
                    << std::endl << std::endl;

                std::exit(EXIT_FAILURE);
            }
#endif
            ~x86_linux_context_impl()
            {
                if (m_stack)
                {
#if defined(HPX_HAVE_VALGRIND) && !defined(NVALGRIND)
                    VALGRIND_STACK_DEREGISTER(
                        reinterpret_cast<std::size_t>(m_sp[valgrind_id_idx]));
#endif
                    posix::free_stack(m_stack, static_cast<std::size_t>(m_stack_size));
                }
            }

            // Return the size of the reserved stack address space.
            std::ptrdiff_t get_stacksize() const
            {
                return m_stack_size;
            }

            void reset_stack()
            {
                if (m_stack)
                {
                    if (posix::reset_stack(
                        m_stack, static_cast<std::size_t>(m_stack_size)))
                        increment_stack_unbind_count();
                }
            }

            void rebind_stack()
            {
                if (m_stack)
                {
                    increment_stack_recycle_count();

                    // On rebind, we initialize our stack to ensure a virgin stack
                    m_sp = (static_cast<void**>(m_stack)
                        + static_cast<std::size_t>(m_stack_size) / sizeof(void*))
                        - context_size;

                    m_sp[cb_idx] = m_sp[backup_cb_idx];
                    m_sp[funp_idx] = m_sp[backup_funp_idx];
                }
            }

            std::ptrdiff_t get_available_stack_space()
            {
                return get_stack_ptr() - reinterpret_cast<std::size_t>(m_stack) -
                    context_size;
            }

            typedef boost::atomic<std::int64_t> counter_type;

            static counter_type& get_stack_unbind_counter()
            {
                static counter_type counter(0);
                return counter;
            }

            static std::uint64_t get_stack_unbind_count(bool reset)
            {
                return util::get_and_reset_value(get_stack_unbind_counter(), reset);
            }

            static std::uint64_t increment_stack_unbind_count()
            {
                return ++get_stack_unbind_counter();
            }

            static counter_type& get_stack_recycle_counter()
            {
                static counter_type counter(0);
                return counter;
            }

            static std::uint64_t get_stack_recycle_count(bool reset)
            {
                return util::get_and_reset_value(get_stack_recycle_counter(), reset);
            }

            static std::uint64_t increment_stack_recycle_count()
            {
                return ++get_stack_recycle_counter();
            }

            friend void swap_context(x86_linux_context_impl_base& from,
                x86_linux_context_impl_base const& to, default_hint);

            friend void swap_context(x86_linux_context_impl_base& from,
                x86_linux_context_impl_base const& to, yield_hint);

            // global functions to be called for each OS-thread after it started
            // running and before it exits
            static void thread_startup(char const* /*thread_type*/)
            {}

            static void thread_shutdown()
            {}

        private:
#if defined(__x86_64__)
            /** structure of context_data:
             * 13: backup address of function to execute
             * 12: backup address of trampoline
             * 11: additional alignment (or valgrind_id if enabled)
             * 10: parm 0 of trampoline
             * 9:  dummy return address for trampoline
             * 8:  return addr (here: start addr)
             * 7:  rbp
             * 6:  rbx
             * 5:  rsi
             * 4:  rdi
             * 3:  r12
             * 2:  r13
             * 1:  r14
             * 0:  r15
             **/
#if defined(HPX_HAVE_VALGRIND) && !defined(NVALGRIND)
            static const std::size_t valgrind_id_idx = 11;
#endif

            static const std::size_t context_size = 14;
            static const std::size_t backup_cb_idx = 13;
            static const std::size_t backup_funp_idx = 12;
            static const std::size_t cb_idx = 10;
            static const std::size_t funp_idx = 8;
#else
            /** structure of context_data:
             * 9: valgrind_id (if enabled)
             * 8: backup address of function to execute
             * 7: backup address of trampoline
             * 6: parm 0 of trampoline
             * 5: dummy return address for trampoline
             * 4: return addr (here: start addr)
             * 3: ebp
             * 2: ebx
             * 1: esi
             * 0: edi
             **/
#if defined(HPX_HAVE_VALGRIND) && !defined(NVALGRIND)
            static const std::size_t context_size = 10;
            static const std::size_t valgrind_id_idx = 9;
#else
            static const std::size_t context_size = 9;
#endif

            static const std::size_t backup_cb_idx = 8;
            static const std::size_t backup_funp_idx = 7;
            static const std::size_t cb_idx = 6;
            static const std::size_t funp_idx = 4;
#endif

            std::ptrdiff_t m_stack_size;
            void* m_stack;

            struct sigaction action;
            stack_t segv_stack;
        };

        typedef x86_linux_context_impl context_impl;

        /**
         * Free function. Saves the current context in @p from
         * and restores the context in @p to.
         * @note This function is found by ADL.
         */
        inline void swap_context(x86_linux_context_impl_base& from,
            x86_linux_context_impl_base const& to, default_hint)
        {
            //        HPX_ASSERT(*(void**)to.m_stack == (void*)~0);
            to.prefetch();
            swapcontext_stack(&from.m_sp, to.m_sp);
        }

        inline void swap_context(x86_linux_context_impl_base& from,
            x86_linux_context_impl_base const& to, yield_hint)
        {
            //        HPX_ASSERT(*(void**)from.m_stack == (void*)~0);
            to.prefetch();
#ifndef HPX_COROUTINE_NO_SEPARATE_CALL_SITES
            swapcontext_stack2(&from.m_sp, to.m_sp);
#else
            swapcontext_stack(&from.m_sp, to.m_sp);
#endif
        }
    }}
}}}

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

#endif /*HPX_RUNTIME_THREADS_COROUTINES_DETAIL_CONTEXT_LINUX_HPP*/
