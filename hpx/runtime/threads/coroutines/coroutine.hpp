//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2007-2013 Hartmut Kaiser
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

#ifndef HPX_RUNTIME_THREADS_COROUTINES_COROUTINE_HPP
#define HPX_RUNTIME_THREADS_COROUTINES_COROUTINE_HPP

#include <hpx/config.hpp>

#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_accessor.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_impl.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_self.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/util/assert.hpp>

#include <boost/mpl/bool.hpp>

#include <cstddef>
#include <limits>
#include <utility>

namespace hpx { namespace threads { namespace coroutines
{
    /////////////////////////////////////////////////////////////////////////////
    class coroutine
    {
    private:
        HPX_MOVABLE_BUT_NOT_COPYABLE(coroutine)

    public:
        friend struct detail::coroutine_accessor;

        typedef thread_state_enum result_type;
        typedef thread_state_ex_enum arg_type;

        typedef detail::coroutine_impl impl_type;
        typedef impl_type::pointer impl_ptr;
        typedef impl_type::thread_id_repr_type thread_id_repr_type;

        typedef util::unique_function_nonser<
            thread_state_enum(thread_state_ex_enum)
        > functor_type;

        coroutine() : m_pimpl(0) {}

        coroutine(functor_type&& f, naming::id_type&& target,
            thread_id_repr_type id = 0, std::ptrdiff_t stack_size =
            detail::default_stack_size)
          : m_pimpl(impl_type::create(
                std::move(f), std::move(target), id, stack_size))
        {
            HPX_ASSERT(m_pimpl->is_ready());
        }

        //coroutine (impl_ptr p)
        //  : m_pimpl(p)
        //{}

        coroutine(coroutine && src)
          : m_pimpl(src.m_pimpl)
        {
            src.m_pimpl = 0;
        }

        coroutine& operator=(coroutine && src)
        {
            coroutine(std::move(src)).swap(*this);
            return *this;
        }

        coroutine& swap(coroutine& rhs)
        {
            std::swap(m_pimpl, rhs.m_pimpl);
            return *this;
        }

        friend void swap(coroutine& lhs, coroutine& rhs)
        {
            lhs.swap(rhs);
        }

        thread_id_repr_type get_thread_id() const
        {
            return m_pimpl->get_thread_id();
        }

#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
        std::size_t get_thread_phase() const
        {
            return m_pimpl->get_thread_phase();
        }
#endif

#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
        std::size_t get_thread_data() const
        {
            return m_pimpl.get() ? m_pimpl->get_thread_data() : 0;
        }

        std::size_t set_thread_data(std::size_t data)
        {
            return m_pimpl.get() ? m_pimpl->set_thread_data(data) : 0;
        }
#endif

        void rebind(functor_type&& f, naming::id_type&& target,
            thread_id_repr_type id = 0)
        {
            HPX_ASSERT(exited());
            impl_type::rebind(m_pimpl.get(), std::move(f),
                std::move(target), id);
        }

        HPX_FORCEINLINE result_type operator()(arg_type arg = arg_type())
        {
            HPX_ASSERT(m_pimpl);
            HPX_ASSERT(m_pimpl->is_ready());

            result_type* ptr = 0;
            m_pimpl->bind_args(&arg);
            m_pimpl->bind_result_pointer(&ptr);

            m_pimpl->invoke();

            return *m_pimpl->result();
        }

        typedef void(coroutine::*bool_type)();
        operator bool_type() const
        {
            return good() ? &coroutine::bool_type_f : 0;
        }

        bool operator==(const coroutine& rhs) const
        {
            return m_pimpl == rhs.m_pimpl;
        }

        void exit()
        {
            HPX_ASSERT(m_pimpl);
            m_pimpl->exit();
        }

        bool waiting() const
        {
            HPX_ASSERT(m_pimpl);
            return m_pimpl->waiting();
        }

        bool pending() const
        {
            HPX_ASSERT(m_pimpl);
            return m_pimpl->pending() != 0;
        }

        bool exited() const
        {
            HPX_ASSERT(m_pimpl);
            return m_pimpl->exited();
        }

        bool is_ready() const
        {
            HPX_ASSERT(m_pimpl);
            return m_pimpl->is_ready();
        }

        bool empty() const
        {
            return m_pimpl == 0;
        }

        std::ptrdiff_t get_available_stack_space()
        {
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
            return m_pimpl->get_available_stack_space();
#else
            return (std::numeric_limits<std::ptrdiff_t>::max)();
#endif
        }

    protected:
        void bool_type_f() {}

        bool good() const
        {
            return !empty() && !exited() && !waiting();
        }

        impl_ptr m_pimpl;

        std::size_t count() const
        {
            return m_pimpl->count();
        }

        impl_ptr get_impl()
        {
            return m_pimpl;
        }
    };
}}}

#endif /*HPX_RUNTIME_THREADS_COROUTINES_COROUTINE_HPP*/
