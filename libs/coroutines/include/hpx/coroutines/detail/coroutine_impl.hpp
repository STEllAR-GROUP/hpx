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
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_COROUTINES_DETAIL_COROUTINE_IMPL_HPP
#define HPX_RUNTIME_THREADS_COROUTINES_DETAIL_COROUTINE_IMPL_HPP

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4355)    //this used in base member initializer
#endif

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/coroutines/coroutine_fwd.hpp>
#include <hpx/coroutines/detail/context_base.hpp>
#include <hpx/coroutines/detail/coroutine_accessor.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/functional/unique_function.hpp>

#include <cstddef>
#include <exception>
#include <utility>

namespace hpx { namespace threads { namespace coroutines { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename ThreadData>
    struct reset_self_on_exit
    {
        reset_self_on_exit(coroutine_self<ThreadData>* val,
            coroutine_self<ThreadData>* old_val = nullptr)
          : old_self(old_val)
        {
            coroutine_self<ThreadData>::set_self(val);
        }

        ~reset_self_on_exit()
        {
            coroutine_self<ThreadData>::set_self(old_self);
        }

        coroutine_self<ThreadData>* old_self;
    };

    ///////////////////////////////////////////////////////////////////////////
    // This type augments the context_base type with the type of the stored
    // functor.
    template <typename ThreadData>
    class coroutine_impl
      : public context_base<coroutine_impl<ThreadData>, ThreadData>
    {
    public:
        HPX_NON_COPYABLE(coroutine_impl);

    public:
        using super_type = context_base<coroutine_impl<ThreadData>, ThreadData>;
        using thread_id_type = typename super_type::thread_id_type;

        using result_type = std::pair<thread_state_enum, thread_id_type>;
        using arg_type = thread_state_ex_enum;

        using functor_type =
            util::unique_function_nonser<result_type(arg_type)>;

        coroutine_impl(
            functor_type&& f, thread_id_type id, std::ptrdiff_t stack_size)
          : super_type(stack_size, id)
          , m_result(unknown, invalid_thread_id)
          , m_arg(nullptr)
          , m_fun(std::move(f))
        {
        }

#if defined(HPX_DEBUG)
        HPX_EXPORT ~coroutine_impl()
        {
            HPX_ASSERT(!m_fun);    // functor should have been reset by now
        }
#endif

        HPX_EXPORT void operator()() noexcept
        {
            using context_exit_status =
                typename super_type::context_exit_status;
            context_exit_status status = super_type::ctx_exited_return;

            // yield value once the thread function has finished executing
            result_type result_last(
                thread_state_enum::terminated, invalid_thread_id);

            // loop as long this coroutine has been rebound
            do
            {
#if defined(HPX_HAVE_ADDRESS_SANITIZER)
                finish_switch_fiber(nullptr, m_caller);
#endif
                std::exception_ptr tinfo;
                try
                {
                    {
                        coroutine_self<ThreadData>* old_self =
                            coroutine_self<ThreadData>::get_self();
                        coroutine_self<ThreadData> self(this, old_self);
                        reset_self_on_exit<ThreadData> on_exit(&self, old_self);

                        result_last = m_fun(*this->args());
                        HPX_ASSERT(
                            result_last.first == thread_state_enum::terminated);
                    }

                    // return value to other side of the fence
                    this->bind_result(result_last);
                }
                catch (...)
                {
                    status = super_type::ctx_exited_abnormally;
                    tinfo = std::current_exception();
                }

                this->reset();
                this->do_return(status, std::move(tinfo));
            } while (this->m_state == super_type::ctx_running);

            // should not get here, never
            HPX_ASSERT(this->m_state == super_type::ctx_running);
        }

    public:
        void bind_result(result_type res)
        {
            m_result = res;
        }

        result_type result() const
        {
            return m_result;
        }
        arg_type* args()
        {
            HPX_ASSERT(m_arg);
            return m_arg;
        };

        void bind_args(arg_type* arg)
        {
            m_arg = arg;
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
            m_fun.reset();    // just reset the bound function
            this->super_type::reset();
        }

        void rebind(functor_type&& f, thread_id_type id)
        {
            this->rebind_stack();    // count how often a coroutines object was reused
            m_fun = std::move(f);
            this->super_type::rebind_base(id);
        }

    private:
        result_type m_result;
        arg_type* m_arg;

        functor_type m_fun;
    };
}}}}    // namespace hpx::threads::coroutines::detail

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif

#endif /*HPX_RUNTIME_THREADS_COROUTINES_DETAIL_COROUTINE_IMPL_HPP*/
