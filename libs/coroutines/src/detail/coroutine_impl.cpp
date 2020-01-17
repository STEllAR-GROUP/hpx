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

#include <hpx/config.hpp>

#include <hpx/assertion.hpp>
#include <hpx/coroutines/coroutine.hpp>
#include <hpx/coroutines/detail/coroutine_accessor.hpp>
#include <hpx/coroutines/detail/coroutine_impl.hpp>
#include <hpx/coroutines/detail/coroutine_self.hpp>

#include <cstddef>
#include <exception>
#include <utility>

namespace hpx { namespace threads { namespace coroutines { namespace detail {
    ///////////////////////////////////////////////////////////////////////////

#if defined(HPX_DEBUG)
    coroutine_impl::~coroutine_impl()
    {
        HPX_ASSERT(!m_fun);    // functor should have been reset by now
    }
#endif

    ////////////////////////////////////////////////////////////////////////////
    void coroutine_impl::operator()() noexcept
    {
        // loop as long this coroutine has been rebound
        do
        {
            context_exit_status status = ctx_not_exited;

            HPX_ASSERT_(!m_fun.empty(), "");

#if defined(HPX_HAVE_ADDRESS_SANITIZER)
            finish_switch_fiber(nullptr, m_caller);
#endif
            std::exception_ptr tinfo;
            {
                basic_execution::agent_ref agent =
                    basic_execution::this_thread::agent();
                coroutine_self* self = coroutine_self::get_self();

                // yield value once the thread function has finished executing
                result_type result_last(
                    thread_state_enum::unknown, invalid_thread_id);

                try
                {
                    result_last = m_fun(*this->args());
                    HPX_ASSERT_(
                        result_last.first == thread_state_enum::terminated, "");
                    status = super_type::ctx_exited_return;
                }
                catch (...)
                {
                    status = super_type::ctx_exited_abnormally;
                    tinfo = std::current_exception();
                }

                this->reset_tss();

                // return value to other side of the fence
                this->bind_result(result_last);
            }

            this->reset();
            this->do_return(status, std::move(tinfo));

        } while (this->m_state == super_type::ctx_running);

        // should not get here, never
        HPX_ASSERT(this->m_state == super_type::ctx_running);
    }

    ////////////////////////////////////////////////////////////////////////////
    // execute the coroutine function directly in the context of the calling
    // thread
    coroutine_impl::result_type coroutine_impl::invoke_directly(
        coroutine_impl::arg_type arg)
    {
        // we can't re-enter a coroutine directly if it is currently yielded
        if (m_exit_status == ctx_exited_yielded)
        {
            // we return 'pending' as the new state to reschedule the thread
            return result_type(thread_state_enum::pending, invalid_thread_id);
        }

        HPX_ASSERT_(m_state == ctx_ready, "");
        m_state = ctx_running;
        m_exit_status = ctx_not_exited;

        context_exit_status status = ctx_not_exited;

        std::exception_ptr tinfo;
        {
            basic_execution::agent_ref agent =
                basic_execution::this_thread::agent();
            coroutine_self* self = coroutine_self::get_self();

            // yield value once the thread function has finished executing
            result_type result_last(
                thread_state_enum::unknown, invalid_thread_id);

            try
            {
                result_last = m_fun(arg);
                HPX_ASSERT_(
                    result_last.first == thread_state_enum::terminated, "");
                status = super_type::ctx_exited_return;
            }
            catch (...)
            {
                status = super_type::ctx_exited_abnormally;
                tinfo = std::current_exception();
            }

            this->reset_tss();
            this->bind_result(result_last);
        }

        this->reset();
        m_state = ctx_exited;
        m_exit_status = status;

        if (tinfo)
        {
            std::rethrow_exception(tinfo);
        }
        return m_result;
    }
}}}}    // namespace hpx::threads::coroutines::detail
