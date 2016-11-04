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
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_COROUTINES_DETAIL_SELF_HPP
#define HPX_RUNTIME_THREADS_COROUTINES_DETAIL_SELF_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_accessor.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_impl.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/function.hpp>

#include <cstddef>
#include <exception>
#include <limits>
#include <utility>

namespace hpx { namespace threads { namespace coroutines { namespace detail
{
    class coroutine_self
    {
        HPX_NON_COPYABLE(coroutine_self);

        // store the current this and write it to the TSS on exit
        struct reset_self_on_exit
        {
            reset_self_on_exit(coroutine_self* self)
              : self_(self)
            {
                set_self(self->next_self_);
            }

            ~reset_self_on_exit()
            {
                set_self(self_);
            }

            coroutine_self* self_;
        };

    public:
        friend struct detail::coroutine_accessor;

        typedef coroutine_impl impl_type;
        typedef impl_type* impl_ptr; // Note, no reference counting here.
        typedef impl_type::thread_id_repr_type thread_id_repr_type;

        typedef impl_type::result_type result_type;
        typedef impl_type::arg_type arg_type;

        typedef util::function_nonser<arg_type(result_type)>
            yield_decorator_type;

        arg_type yield(result_type arg = result_type())
        {
            return !yield_decorator_.empty() ?
                yield_decorator_(std::move(arg)) :
                yield_impl(std::move(arg));
        }

        arg_type yield_impl(result_type arg)
        {
            HPX_ASSERT(m_pimpl);

            this->m_pimpl->bind_result(&arg);

            {
                reset_self_on_exit on_exit(this);
                this->m_pimpl->yield();
            }

            return *m_pimpl->args();
        }

        template <typename F>
        yield_decorator_type decorate_yield(F && f)
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

        yield_decorator_type decorate_yield(yield_decorator_type && f)
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

        HPX_ATTRIBUTE_NORETURN void exit()
        {
            m_pimpl->exit_self();
            std::terminate(); // FIXME: replace with hpx::terminate();
        }

        bool pending() const
        {
            HPX_ASSERT(m_pimpl);
            return m_pimpl->pending() != 0;
        }

        thread_id_repr_type get_thread_id() const
        {
            HPX_ASSERT(m_pimpl);
            return m_pimpl->get_thread_id();
        }

        std::size_t get_thread_phase() const
        {
#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
            HPX_ASSERT(m_pimpl);
            return m_pimpl->get_thread_phase();
#else
            return 0;
#endif
        }

        std::ptrdiff_t get_available_stack_space()
        {
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
            return m_pimpl->get_available_stack_space();
#else
            return (std::numeric_limits<std::ptrdiff_t>::max)();
#endif
        }

        explicit coroutine_self(impl_type * pimpl,
                coroutine_self* next_self = nullptr)
          : m_pimpl(pimpl), next_self_(next_self)
        {}

        std::size_t get_thread_data() const
        {
            HPX_ASSERT(m_pimpl);
            return m_pimpl->get_thread_data();
        }
        std::size_t set_thread_data(std::size_t data)
        {
            HPX_ASSERT(m_pimpl);
            return m_pimpl->set_thread_data(data);
        }

#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
        tss_storage* get_thread_tss_data()
        {
            HPX_ASSERT(m_pimpl);
            return m_pimpl->get_thread_tss_data(false);
        }

        tss_storage* get_or_create_thread_tss_data()
        {
            HPX_ASSERT(m_pimpl);
            return m_pimpl->get_thread_tss_data(true);
        }
#endif

        std::size_t& get_continuation_recursion_count()
        {
            HPX_ASSERT(m_pimpl);
            return m_pimpl->get_continuation_recursion_count();
        }

    public:
        static HPX_EXPORT void set_self(coroutine_self* self);
        static HPX_EXPORT coroutine_self* get_self();
        static HPX_EXPORT void init_self();
        static HPX_EXPORT void reset_self();

    private:
        yield_decorator_type yield_decorator_;

        impl_ptr get_impl()
        {
            return m_pimpl;
        }
        impl_ptr m_pimpl;
        coroutine_self* next_self_;
    };
}}}}

#endif /*HPX_RUNTIME_THREADS_COROUTINES_DETAIL_SELF_HPP*/
