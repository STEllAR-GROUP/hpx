//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2007-2023 Hartmut Kaiser
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

#include <hpx/config.hpp>

#include <hpx/assert.hpp>
#include <hpx/coroutines/coroutine_fwd.hpp>
#include <hpx/coroutines/detail/coroutine_accessor.hpp>
#include <hpx/coroutines/detail/coroutine_impl.hpp>
#include <hpx/coroutines/detail/coroutine_self.hpp>
#include <hpx/coroutines/thread_id_type.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace hpx::threads::coroutines {

    ///////////////////////////////////////////////////////////////////////////
    class coroutine
    {
    private:
        friend struct detail::coroutine_accessor;

    public:
        using impl_type = detail::coroutine_impl;
        using thread_id_type = impl_type::thread_id_type;

        using result_type = impl_type::result_type;
        using arg_type = impl_type::arg_type;

        using functor_type = hpx::move_only_function<result_type(arg_type)>;

        coroutine(functor_type&& f, thread_id_type id,
            std::ptrdiff_t stack_size = detail::default_stack_size)
          : impl_(HPX_MOVE(f), HPX_MOVE(id), stack_size)
        {
            HPX_ASSERT(impl_.is_ready());
        }

        coroutine(coroutine const& src) = delete;
        coroutine& operator=(coroutine const& src) = delete;
        coroutine(coroutine&& src) = delete;
        coroutine& operator=(coroutine&& src) = delete;

        ~coroutine() = default;

        constexpr thread_id_type get_thread_id() const noexcept
        {
            return impl_.get_thread_id();
        }

#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
        constexpr std::size_t get_thread_phase() const noexcept
        {
            return impl_.get_thread_phase();
        }
#endif

        constexpr std::size_t get_thread_data() const noexcept
        {
            return impl_.get_thread_data();
        }

        std::size_t set_thread_data(std::size_t data) const noexcept
        {
            return impl_.set_thread_data(data);
        }

#if defined(HPX_HAVE_LIBCDS)
        std::size_t get_libcds_data() const
        {
            return impl_.get_libcds_data();
        }

        std::size_t set_libcds_data(std::size_t data) const
        {
            return impl_.set_libcds_data(data);
        }

        std::size_t get_libcds_hazard_pointer_data() const
        {
            return impl_.get_libcds_hazard_pointer_data();
        }

        std::size_t set_libcds_hazard_pointer_data(std::size_t data) const
        {
            return impl_.set_libcds_hazard_pointer_data(data);
        }

        std::size_t get_libcds_dynamic_hazard_pointer_data() const
        {
            return impl_.get_libcds_dynamic_hazard_pointer_data();
        }

        std::size_t set_libcds_dynamic_hazard_pointer_data(
            std::size_t data) const
        {
            return impl_.set_libcds_dynamic_hazard_pointer_data(data);
        }
#endif

        void init()
        {
            impl_.init();
        }

        void rebind(functor_type&& f, thread_id_type id)
        {
            impl_.rebind(HPX_MOVE(f), HPX_MOVE(id));
        }

        HPX_FORCEINLINE result_type operator()(arg_type arg = arg_type())
        {
            HPX_ASSERT(impl_.is_ready());

            impl_.bind_args(&arg);

            impl_.invoke();

            return impl_.result();
        }

        HPX_FORCEINLINE result_type invoke_directly(arg_type arg = arg_type())
        {
            HPX_ASSERT(impl_.is_ready());
            return impl_.invoke_directly(arg);
        }

        bool is_ready() const noexcept
        {
            return impl_.is_ready();
        }

#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
        std::ptrdiff_t get_available_stack_space() const noexcept
        {
            return impl_.get_available_stack_space();
        }
#else
        static std::ptrdiff_t get_available_stack_space() noexcept
        {
            return (std::numeric_limits<std::ptrdiff_t>::max)();
        }
#endif

        constexpr impl_type* impl() noexcept
        {
            return &impl_;
        }

    private:
        impl_type impl_;
    };
}    // namespace hpx::threads::coroutines
