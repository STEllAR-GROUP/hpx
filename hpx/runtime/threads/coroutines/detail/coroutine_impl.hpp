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

#ifndef HPX_RUNTIME_THREADS_COROUTINES_DETAIL_COROUTINE_IMPL_HPP
#define HPX_RUNTIME_THREADS_COROUTINES_DETAIL_COROUTINE_IMPL_HPP

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning (push)
#pragma warning (disable: 4355) //this used in base member initializer
#endif

#include <hpx/config.hpp>
#include <hpx/runtime/threads/coroutines/coroutine_fwd.hpp>
#include <hpx/runtime/threads/coroutines/detail/context_base.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_accessor.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/unique_function.hpp>

#include <boost/exception_ptr.hpp>
#include <boost/intrusive_ptr.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace threads { namespace coroutines { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // This type augments the context_base type with the type of the stored
    // functor.
    class coroutine_impl
      : public context_base
    {
        HPX_NON_COPYABLE(coroutine_impl);

    public:
        typedef context_base super_type;
        typedef context_base::thread_id_repr_type thread_id_repr_type;
        typedef boost::intrusive_ptr<threads::thread_data> thread_id_type;

        typedef std::pair<thread_state_enum, thread_id_type> result_type;
        typedef thread_state_ex_enum arg_type;

        typedef util::unique_function_nonser<result_type(arg_type)> functor_type;

        typedef boost::intrusive_ptr<coroutine_impl> pointer;

        coroutine_impl(functor_type&& f, thread_id_repr_type id,
            std::ptrdiff_t stack_size)
          : context_base(*this, stack_size, id)
          , m_result_last(std::make_pair(thread_state_enum::unknown, nullptr))
          , m_arg(nullptr)
          , m_result(nullptr)
          , m_fun(std::move(f))
        {}

        ~coroutine_impl()
#if defined(HPX_DEBUG)
            ;
#else
        {}
#endif

        static inline coroutine_impl* create(
            functor_type&& f, thread_id_repr_type id = nullptr,
            std::ptrdiff_t stack_size = default_stack_size)
        {
            coroutine_impl* p = allocate(id, stack_size);

            if (!p)
            {
                std::size_t const heap_num = std::size_t(id) / 32; //-V112

                // allocate a new coroutine object, if non is available (or all
                // heaps are locked)
                context_base::increment_allocation_count(heap_num);
                p = new coroutine_impl(std::move(f),
                    id, stack_size);
            } else {
                // we reuse an existing object, we need to rebind its function
                p->rebind(std::move(f), id);
            }
            return p;
        }

        static inline void rebind(
            coroutine_impl* p, functor_type&& f, thread_id_repr_type id = nullptr)
        {
            p->rebind(std::move(f), id);
        }

        static inline void destroy(coroutine_impl* p)
        {
            // always hand the stack back to the matching heap
            deallocate(p);
        }

        HPX_EXPORT void operator()();

    public:
        result_type * result()
        {
            HPX_ASSERT(m_result);
            return *this->m_result;
        }

        arg_type * args()
        {
            HPX_ASSERT(m_arg);
            return m_arg;
        };

        void bind_args(arg_type* arg)
        {
            m_arg = arg;
        }

        void bind_result(result_type* res)
        {
            *m_result = res;
        }

        // Another level of indirection is needed to handle
        // yield_to correctly.
        void bind_result_pointer(result_type** resp)
        {
            m_result = resp;
        }

#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
        std::size_t get_thread_phase() const
        {
            return this->phase();
        }
#endif

        void reset()
        {
            this->reset_stack();
            m_fun.reset(); // just reset the bound function
            this->super_type::reset();
        }

        void rebind(functor_type && f, thread_id_repr_type id)
        {
            this->rebind_stack();     // count how often a coroutines object was reused
            m_fun = std::move(f);
            this->super_type::rebind_base(id);
        }

    private:
        static HPX_EXPORT coroutine_impl* allocate(
            thread_id_repr_type id, std::ptrdiff_t stacksize);

        static HPX_EXPORT void deallocate(coroutine_impl* wrapper);

    private:
        result_type m_result_last;
        arg_type* m_arg;
        result_type** m_result;

        functor_type m_fun;
    };
}}}}

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif

#endif /*HPX_RUNTIME_THREADS_COROUTINES_DETAIL_COROUTINE_IMPL_HPP*/
