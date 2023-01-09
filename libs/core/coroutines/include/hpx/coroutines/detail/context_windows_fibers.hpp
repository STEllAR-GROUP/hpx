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
#include <hpx/coroutines/detail/swap_context.hpp>
#include <hpx/util/get_and_reset_value.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <system_error>

#if defined(HPX_HAVE_ADDRESS_SANITIZER)
#include <processthreadsapi.h>
#include <sanitizer/asan_interface.h>
#endif

#if defined(HPX_COROUTINES_HAVE_SWAP_CONTEXT_EMULATION)
extern "C" void switch_to_fiber(void* lpFiber) noexcept;
#endif

namespace hpx::threads::coroutines {

    // On Windows we need a special preparation for the main coroutines thread
    struct prepare_main_thread
    {
        prepare_main_thread() noexcept
        {
            [[maybe_unused]] LPVOID const result =
                ConvertThreadToFiber(nullptr);
            HPX_ASSERT(nullptr != result);
        }

        HPX_NON_COPYABLE(prepare_main_thread);

        ~prepare_main_thread() noexcept
        {
            [[maybe_unused]] BOOL const result = ConvertFiberToThread();
            HPX_ASSERT(FALSE != result);
        }
    };

    namespace detail::windows {

        using fiber_ptr = LPVOID;

        // Return true if current thread is a fiber.
        inline bool is_fiber() noexcept
        {
#if _WIN32_WINNT >= 0x0600
            return IsThreadAFiber() ? true : false;
#else
            // This number (0x1E00) has been sighted in the wild (at least on
            // windows XP systems) as return value from GetCurrentFiber() on non
            // fibrous threads.
            // This is somehow related to OS/2 where the current fiber pointer is
            // overloaded as a version field.
            // On non-NT systems, 0 is returned.
            static fiber_ptr const fiber_magic =
                reinterpret_cast<fiber_ptr>(0x1E00);

            fiber_ptr current = GetCurrentFiber();
            return current != nullptr && current != fiber_magic;
#endif
        }

        // Windows implementation for the context_impl_base class.
        //
        // @note context_impl is not required to be consistent. If not
        //       initialized it can only be swapped out, not in (at that point
        //       it will be initialized).
        class fibers_context_impl_base : detail::context_impl_base
        {
        public:
            // Create an empty context. An empty context cannot be restored
            // from, but can be saved in.
            fibers_context_impl_base() = default;

            // Free function. Saves the current context in @p from and restores
            // the context in @p to. On windows the from parameter is ignored.
            // The current context is saved on the current fiber.
            //
            // Note that if the current thread is not a fiber, it will be
            // converted to fiber on the fly on call and unconverted before
            // return. This is expensive. The user should convert the current
            // thread to a fiber once on thread creation for better performance.
            //
            // Note that we can't leave the thread unconverted on return or else
            // we will leak resources on thread destruction. Do the right thing
            // by default.
            friend void swap_context(fibers_context_impl_base& from,
                fibers_context_impl_base const& to, default_hint) noexcept
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
                    [[maybe_unused]] BOOL const result = ConvertFiberToThread();
                    HPX_ASSERT(result);
                    from.m_ctx = nullptr;
                }
                else
                {
                    bool const call_from_main = from.m_ctx == nullptr;
                    if (call_from_main)
                        from.m_ctx = GetCurrentFiber();
                    HPX_ASSERT(from.m_ctx != nullptr);

#if defined(HPX_COROUTINES_HAVE_SWAP_CONTEXT_EMULATION)
                    switch_to_fiber(to.m_ctx);
#else
                    SwitchToFiber(to.m_ctx);
#endif
                    if (call_from_main)
                        from.m_ctx = nullptr;
                }
            }

            HPX_NON_COPYABLE(fibers_context_impl_base);

            ~fibers_context_impl_base() = default;

#if defined(HPX_HAVE_ADDRESS_SANITIZER)
            void start_switch_fiber(void** fake_stack) noexcept
            {
                if (asan_stack_bottom == nullptr)
                {
                    void const* dummy = nullptr;
                    GetCurrentThreadStackLimits(
                        (PULONG_PTR) &asan_stack_bottom, (PULONG_PTR) &dummy);
                }
                __sanitizer_start_switch_fiber(
                    fake_stack, asan_stack_bottom, asan_stack_size);
            }

            void start_yield_fiber(
                void** fake_stack, fibers_context_impl_base& caller) noexcept
            {
                __sanitizer_start_switch_fiber(fake_stack,
                    caller.asan_stack_bottom, caller.asan_stack_size);
            }

            void finish_yield_fiber(void* fake_stack) noexcept
            {
                __sanitizer_finish_switch_fiber(
                    fake_stack, &asan_stack_bottom, &asan_stack_size);
            }

            void finish_switch_fiber(
                void* fake_stack, fibers_context_impl_base& caller)
            {
                __sanitizer_finish_switch_fiber(fake_stack,
                    &caller.asan_stack_bottom, &caller.asan_stack_size);
            }
#endif

        protected:
            explicit constexpr fibers_context_impl_base(fiber_ptr ctx) noexcept
              : m_ctx(ctx)
            {
            }

            fiber_ptr m_ctx = nullptr;

#if defined(HPX_HAVE_ADDRESS_SANITIZER)
        public:
            void* asan_fake_stack = nullptr;
            void const* asan_stack_bottom = nullptr;
            std::size_t asan_stack_size = 0;
#endif
        };

        template <typename T>
        void CALLBACK trampoline(LPVOID pv)
        {
            T* fun = static_cast<T*>(pv);
            HPX_ASSERT(fun);
            (*fun)();
        }

        // initial stack size (grows as needed)
        inline constexpr std::size_t stack_size =
            sizeof(void*) >= 8 ? 2048 : 1024;

        template <typename CoroutineImpl>
        class fibers_context_impl : public fibers_context_impl_base
        {
        public:
            HPX_NON_COPYABLE(fibers_context_impl);

        public:
            using context_impl_base = fibers_context_impl_base;

            static constexpr std::size_t default_stack_size = stack_size;

            // Create a context that on restore invokes Functor on a new stack.
            // The stack size can be optionally specified.
            explicit fibers_context_impl(std::ptrdiff_t stacksize) noexcept
              : stacksize_(stacksize == -1 ?
                        static_cast<std::ptrdiff_t>(default_stack_size) :
                        stacksize)
            {
            }

            void init()
            {
                if (m_ctx != nullptr)
                    return;

                m_ctx = CreateFiberEx(stacksize_, stacksize_, 0,
                    static_cast<LPFIBER_START_ROUTINE>(
                        &trampoline<CoroutineImpl>),
                    static_cast<LPVOID>(this));
                if (nullptr == m_ctx)
                {
                    throw std::system_error(static_cast<int>(GetLastError()),
                        std::system_category());
                }

#if defined(HPX_HAVE_ADDRESS_SANITIZER)
                this->asan_stack_size = stacksize_;
                this->asan_stack_bottom = nullptr;
#endif
            }

            ~fibers_context_impl()
            {
                if (m_ctx != nullptr)
                    DeleteFiber(m_ctx);
            }

            // Return the size of the reserved stack address space.
            [[nodiscard]] constexpr std::ptrdiff_t get_stacksize()
                const noexcept
            {
                return stacksize_;
            }

            static constexpr void reset_stack() noexcept {}

#if defined(HPX_HAVE_COROUTINE_COUNTERS)
            void rebind_stack() noexcept
            {
                increment_stack_recycle_count();
            }
#else
            static constexpr void rebind_stack() noexcept {}
#endif

            // Detect remaining stack space (approximate), taken from here:
            // https://stackoverflow.com/a/20930496/269943
            static std::ptrdiff_t get_available_stack_space() noexcept
            {
                MEMORY_BASIC_INFORMATION mbi = {};        // page range
                VirtualQuery(&mbi, &mbi, sizeof(mbi));    // get range
                return reinterpret_cast<std::ptrdiff_t>(&mbi) -
                    reinterpret_cast<std::ptrdiff_t>(mbi.AllocationBase);
            }

#if defined(HPX_HAVE_COROUTINE_COUNTERS)
            using counter_type = std::atomic<std::int64_t>;

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
    }    // namespace detail::windows
}    // namespace hpx::threads::coroutines
