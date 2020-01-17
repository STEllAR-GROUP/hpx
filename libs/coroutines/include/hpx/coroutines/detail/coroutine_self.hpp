//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2007-2020, Hartmut Kaiser
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

#ifndef HPX_RUNTIME_THREADS_COROUTINES_DETAIL_SELF_HPP
#define HPX_RUNTIME_THREADS_COROUTINES_DETAIL_SELF_HPP

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/basic_execution.hpp>
#include <hpx/coroutines/detail/coroutine_accessor.hpp>
#include <hpx/coroutines/detail/coroutine_impl.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/functional/function.hpp>

#include <cstddef>
#include <exception>
#include <limits>
#include <utility>

namespace hpx { namespace threads { namespace coroutines { namespace detail {

    class coroutine_self
    {
    public:
        HPX_NON_COPYABLE(coroutine_self);

    public:
        using thread_id_type = hpx::threads::thread_id;

        using result_type = std::pair<thread_state_enum, thread_id_type>;
        using arg_type = thread_state_ex_enum;

        using yield_decorator_type =
            util::function_nonser<arg_type(result_type)>;

        explicit coroutine_self(
            coroutine_impl* pimpl, coroutine_impl* next_impl = nullptr)
          : pimpl_(pimpl)
          , pimpl_next_(next_impl)
        {
        }

        arg_type yield(result_type arg = result_type())
        {
            return !yield_decorator_.empty() ?
                yield_decorator_(std::move(arg)) :
                yield_impl(std::move(arg));
        }

        template <typename F>
        yield_decorator_type decorate_yield(F&& f)
        {
            yield_decorator_type tmp(std::forward<F>(f));
            std::swap(tmp, yield_decorator_);
            return tmp;
        }

        yield_decorator_type decorate_yield(yield_decorator_type const& f)
        {
            yield_decorator_type tmp(f);
            std::swap(tmp, yield_decorator_);
            return tmp;
        }

        yield_decorator_type decorate_yield(yield_decorator_type&& f)
        {
            std::swap(f, yield_decorator_);
            return std::move(f);
        }

        yield_decorator_type undecorate_yield()
        {
            yield_decorator_type tmp;
            std::swap(tmp, yield_decorator_);
            return tmp;
        }

        ~coroutine_self() = default;

    private:
        HPX_FORCEINLINE arg_type do_yield(coroutine_impl* coro, result_type arg)
        {
            HPX_ASSERT(coro);

            coro->bind_result(arg);
            coro->yield();
            return *coro->args();
        }

#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
        HPX_FORCEINLINE std::size_t do_get_thread_phase(
            coroutine_impl* coro) const
        {
            HPX_ASSERT(coro);
            return coro->get_thread_phase();
        }
#endif

#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
        HPX_FORCEINLINE std::size_t do_get_available_stack_space(
            coroutine_impl* coro) const
        {
            HPX_ASSERT(coro);
            return coro->get_available_stack_space();
        }
#endif

        HPX_FORCEINLINE std::size_t& do_get_continuation_recursion_count(
            coroutine_impl* coro)
        {
            HPX_ASSERT(coro);
            return coro->get_continuation_recursion_count();
        }

        HPX_FORCEINLINE thread_id_type do_get_thread_id(
            coroutine_impl* coro) const
        {
            HPX_ASSERT(coro);
            return coro->get_thread_id();
        }

    public:
        arg_type yield_impl(result_type arg)
        {
            return do_yield(pimpl_next_ != nullptr ? pimpl_next_ : pimpl_, arg);
        }

        std::size_t get_thread_phase() const
        {
#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
            return do_get_thread_phase(
                pimpl_next_ != nullptr ? pimpl_next_ : pimpl_);
#else
            return 0;
#endif
        }

        std::ptrdiff_t get_available_stack_space()
        {
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
            return do_get_available_stack_space(
                pimpl_next_ != nullptr ? pimpl_next_ : pimpl_);
#else
            return (std::numeric_limits<std::ptrdiff_t>::max)();
#endif
        }

        thread_id_type get_thread_id() const
        {
            return do_get_thread_id(
                pimpl_next_ != nullptr ? pimpl_next_ : pimpl_);
        }

        std::size_t& get_continuation_recursion_count()
        {
            return do_get_continuation_recursion_count(
                pimpl_next_ != nullptr ? pimpl_next_ : pimpl_);
        }

        std::size_t get_thread_data() const
        {
            HPX_ASSERT(pimpl_);
            return pimpl_->get_thread_data();
        }
        std::size_t set_thread_data(std::size_t data)
        {
            HPX_ASSERT(pimpl_);
            return pimpl_->set_thread_data(data);
        }

        tss_storage* get_thread_tss_data()
        {
#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
            HPX_ASSERT(pimpl_);
            return pimpl_->get_thread_tss_data(false);
#else
            return nullptr;
#endif
        }

        tss_storage* get_or_create_thread_tss_data()
        {
#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
            HPX_ASSERT(pimpl_);
            return pimpl_->get_thread_tss_data(true);
#else
            return nullptr;
#endif
        }

    public:
        static coroutine_self* get_self()
        {
            return basic_execution::this_thread::get_agent_data<coroutine_self>(
                basic_execution::this_thread::agent());
        }

        static void set_self(coroutine_self* new_self)
        {
            coroutine_self* self = get_self();
            HPX_ASSERT(self);

//             if (new_self != nullptr)
//             {
//                 self->set_impls(new_self->get_impls());
//             }
//             else
//             {
//                 self->set_impls(nullptr, nullptr);
//             }
        }

        // access coroutines context object
        using impl_type = coroutine_impl;
        using impl_ptr = coroutine_impl*;

    private:
        friend struct coroutine_accessor;

        HPX_FORCEINLINE coroutine_impl* get_impl()
        {
            return pimpl_;
        }

        HPX_FORCEINLINE void set_impl(coroutine_impl* coro)
        {
            pimpl_ = coro;
        }

        HPX_FORCEINLINE coroutine_impl* get_impl_next()
        {
            return pimpl_next_;
        }

        HPX_FORCEINLINE void set_impl_next(coroutine_impl* coro)
        {
            pimpl_next_ = coro;
        }

//         HPX_FORCEINLINE std::pair<coroutine_impl*, coroutine_impl*> get_impls()
//             const noexcept
//         {
//             return std::make_pair(pimpl_, pimpl_next_);
//         }
//
//         HPX_FORCEINLINE void set_impls(
//             std::pair<coroutine_impl*, coroutine_impl*> new_impl) noexcept
//         {
//             pimpl_ = new_impl.first;
//             pimpl_next_ = new_impl.second;
//         }
//
//         HPX_FORCEINLINE void set_impls(
//             coroutine_impl* coro, coroutine_impl* next) noexcept
//         {
//             pimpl_ = coro;
//             pimpl_next_ = next;
//         }

    private:
        yield_decorator_type yield_decorator_;
        coroutine_impl* pimpl_;
        coroutine_impl* pimpl_next_;
    };

}}}}    // namespace hpx::threads::coroutines::detail

#endif
