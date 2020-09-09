//  Copyright (c) 2006, Giovanni P. Deretta
//
//  This code may be used under either of the following two licences:
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE. OF SUCH DAMAGE.
//
//  Or:
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <windows.h>
#include <winnt.h>

#include <hpx/config.hpp>
#include <hpx/coroutines/config/defines.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/coroutines/detail/swap_context.hpp>
#include <hpx/type_support/unused.hpp>
#include <hpx/util/get_and_reset_value.hpp>

#include <boost/system/error_code.hpp>
#include <boost/system/system_error.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>

#if defined(HPX_COROUTINES_HAVE_SWAP_CONTEXT_EMULATION)
extern "C" void switch_to_fiber(void* lpFiber) noexcept;
#endif

namespace hpx { namespace threads { namespace coroutines {
    // On Windows we need a special preparation for the main coroutines thread
    struct prepare_main_thread
    {
        prepare_main_thread() noexcept
        {
            LPVOID result = ConvertThreadToFiber(nullptr);
            HPX_ASSERT(nullptr != result);
            HPX_UNUSED(result);
        }

        ~prepare_main_thread() noexcept
        {
            BOOL result = ConvertFiberToThread();
            HPX_ASSERT(FALSE != result);
            HPX_UNUSED(result);
        }
    };

    namespace detail { namespace windows {
        typedef LPVOID fiber_ptr;

#if _WIN32_WINNT < 0x0600
        /*
         * This number (0x1E00) has been sighted in the wild (at least on
         * windows XP systems) as return value from GetCurrentFiber() on non
         * fibrous threads.
         * This is somehow related to OS/2 where the current fiber pointer is
         * overloaded as a version field.
         * On non-NT systems, 0 is returned.
         */
        fiber_ptr const fiber_magic = reinterpret_cast<fiber_ptr>(0x1E00);
#endif

        /*
         * Return true if current thread is a fiber.
         */
        inline bool is_fiber() noexcept
        {
#if _WIN32_WINNT >= 0x0600
            return IsThreadAFiber() ? true : false;
#else
            fiber_ptr current = GetCurrentFiber();
            return current != nullptr && current != fiber_magic;
#endif
        }

        /*
         * Windows implementation for the context_impl_base class.
         * @note context_impl is not required to be consistent
         * If not initialized it can only be swapped out, not in
         * (at that point it will be initialized).
         */
        class fibers_context_impl_base : detail::context_impl_base
        {
        public:
            /**
             * Create an empty context.
             * An empty context cannot be restored from,
             * but can be saved in.
             */
            fibers_context_impl_base() noexcept
              : m_ctx(nullptr)
            {
            }

            /*
             * Free function. Saves the current context in @p from
             * and restores the context in @p to. On windows the from
             * parameter is ignored. The current context is saved on the
             * current fiber.
             * Note that if the current thread is not a fiber, it will be
             * converted to fiber on the fly on call and unconverted before
             * return. This is expensive. The user should convert the
             * current thread to a fiber once on thread creation for better performance.
             * Note that we can't leave the thread unconverted on return or else we
             * will leak resources on thread destruction. Do the right thing by
             * default.
             */
            friend void swap_context(fibers_context_impl_base& from,
                const fibers_context_impl_base& to, default_hint) noexcept
            {
                if (!is_fiber())
                {
                    HPX_ASSERT(from.m_ctx == nullptr);
                    from.m_ctx = ConvertThreadToFiber(nullptr);
                    HPX_ASSERT(from.m_ctx != nullptr);

#if defined(HPX_COROUTINES_HAVE_SWAP_CONTEXT_EMULATION)
                    switch_to_fiber(to.m_ctx);
#else
                    SwitchToFiber(to.m_ctx);
#endif
                    BOOL result = ConvertFiberToThread();
                    HPX_ASSERT(result);
                    HPX_UNUSED(result);
                    from.m_ctx = nullptr;
                }
                else
                {
                    bool call_from_main = from.m_ctx == nullptr;
                    if (call_from_main)
                        from.m_ctx = GetCurrentFiber();
#if defined(HPX_COROUTINES_HAVE_SWAP_CONTEXT_EMULATION)
                    switch_to_fiber(to.m_ctx);
#else
                    SwitchToFiber(to.m_ctx);
#endif
                    if (call_from_main)
                        from.m_ctx = nullptr;
                }
            }

            ~fibers_context_impl_base() {}

        protected:
            explicit fibers_context_impl_base(fiber_ptr ctx) noexcept
              : m_ctx(ctx)
            {
            }

            fiber_ptr m_ctx;
        };

        template <typename T>
        HPX_FORCEINLINE VOID CALLBACK trampoline(LPVOID pv)
        {
            T* fun = static_cast<T*>(pv);
            HPX_ASSERT(fun);
            (*fun)();
        }

        // initial stack size (grows as needed)
        static const std::size_t stack_size = sizeof(void*) >= 8 ? 2048 : 1024;

        template <typename CoroutineImpl>
        class fibers_context_impl : public fibers_context_impl_base
        {
        public:
            HPX_NON_COPYABLE(fibers_context_impl);

        public:
            typedef fibers_context_impl_base context_impl_base;

            enum
            {
                default_stack_size = stack_size
            };

            /**
             * Create a context that on restore invokes Functor on
             *  a new stack. The stack size can be optionally specified.
             */
            explicit fibers_context_impl(std::ptrdiff_t stack_size)
              : stacksize_(stack_size == -1 ?
                        std::ptrdiff_t(default_stack_size) :
                        stack_size)
            {
            }

            void init()
            {
                if (m_ctx != nullptr)
                    return;

                m_ctx = CreateFiberEx(stack_size, stack_size, 0,
                    static_cast<LPFIBER_START_ROUTINE>(
                        &trampoline<CoroutineImpl>),
                    static_cast<LPVOID>(this));
                if (nullptr == m_ctx)
                {
                    throw boost::system::system_error(boost::system::error_code(
                        GetLastError(), boost::system::system_category()));
                }
            }

            ~fibers_context_impl()
            {
                if (m_ctx != nullptr)
                    DeleteFiber(m_ctx);
            }

            // Return the size of the reserved stack address space.
            std::ptrdiff_t get_stacksize() const noexcept
            {
                return stacksize_;
            }

            constexpr void reset_stack() noexcept {}

            void rebind_stack() noexcept
            {
#if defined(HPX_HAVE_COROUTINE_COUNTERS)
                increment_stack_recycle_count();
#endif
            }

            // Detect remaining stack space (approximate), taken from here:
            // https://stackoverflow.com/a/20930496/269943
            std::ptrdiff_t get_available_stack_space()
            {
                MEMORY_BASIC_INFORMATION mbi;                     // page range
                VirtualQuery((PVOID) &mbi, &mbi, sizeof(mbi));    // get range
                return (std::ptrdiff_t) &mbi -
                    (std::ptrdiff_t) mbi.AllocationBase;
            }

#if defined(HPX_HAVE_COROUTINE_COUNTERS)
            typedef std::atomic<std::int64_t> counter_type;

        private:
            static counter_type& get_stack_recycle_counter() noexcept
            {
                static counter_type counter(0);
                return counter;
            }

            static std::uint64_t increment_stack_recycle_count() noexcept
            {
                return ++get_stack_recycle_counter();
            }

        public:
            static std::uint64_t get_stack_recycle_count(bool reset) noexcept
            {
                return util::get_and_reset_value(
                    get_stack_recycle_counter(), reset);
            }
#endif

        private:
            std::ptrdiff_t stacksize_;
        };
    }}    // namespace detail::windows
}}}       // namespace hpx::threads::coroutines
