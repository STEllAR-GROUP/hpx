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

#ifndef HPX_RUNTIME_THREADS_COROUTINES_DETAIL_CONTEXT_BASE_HPP
#define HPX_RUNTIME_THREADS_COROUTINES_DETAIL_CONTEXT_BASE_HPP

/*
 * Currently asio can, in some cases. call copy constructors and
 * operator= from different threads, even if in the
 * one-thread-per-service model. (i.e. from the resolver thread)
 * This will be corrected in future versions, but for now
 * we will play it safe and use an atomic count. The overhead shouldn't
 * be big.
 */
#include <hpx/config.hpp>

// This needs to be first for building on Macs
#include <hpx/runtime/threads/coroutines/detail/context_impl.hpp>

#include <hpx/runtime/threads/coroutines/detail/swap_context.hpp> //for swap hints
#include <hpx/runtime/threads/coroutines/detail/tss.hpp>
#include <hpx/runtime/threads/thread_id_type.hpp>
#include <hpx/util/assert.hpp>
#if defined(HPX_HAVE_APEX)
#include <hpx/util/apex.hpp>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>


///////////////////////////////////////////////////////////////////////////////
#define HPX_COROUTINE_NUM_ALL_HEAPS (HPX_COROUTINE_NUM_HEAPS +                \
    HPX_COROUTINE_NUM_HEAPS/2 + HPX_COROUTINE_NUM_HEAPS/4 +                   \
    HPX_COROUTINE_NUM_HEAPS/4)                                                \
/**/

namespace hpx { namespace threads { namespace coroutines { namespace detail
{
    //////////////////////////////////////////////////////////////////////////
    //
    struct allocation_counters
    {
        allocation_counters()
        {
            for (std::size_t i = 0; i < HPX_COROUTINE_NUM_ALL_HEAPS; ++i)
            {
                m_allocation_counter[i].store(0);
            }
        }

        std::atomic<std::uint64_t>& get(std::size_t i)
        {
            return m_allocation_counter[i % HPX_COROUTINE_NUM_ALL_HEAPS];
        }

        std::atomic<std::uint64_t> m_allocation_counter[HPX_COROUTINE_NUM_ALL_HEAPS];
    };

    /////////////////////////////////////////////////////////////////////////////
    std::ptrdiff_t const default_stack_size = -1;

#if defined(HPX_HAVE_APEX)
    apex_task_wrapper rebind_base_apex(thread_id_type id);
#endif

    class context_base : public default_context_impl
    {
    public:
        typedef void deleter_type(context_base const*);
        typedef hpx::threads::thread_id_type thread_id_type;

        template <typename Derived>
        context_base(
            Derived& derived, std::ptrdiff_t stack_size, thread_id_type id)
          : default_context_impl(derived, stack_size)
          , m_caller()
          , m_state(ctx_ready)
          , m_exit_state(ctx_exit_not_requested)
          , m_exit_status(ctx_not_exited)
#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
          , m_phase(0)
#endif
#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
          , m_thread_data(nullptr)
#else
          , m_thread_data(0)
#endif
#if defined(HPX_HAVE_APEX)
          , m_apex_data(nullptr)
#endif
          , m_type_info()
          , m_thread_id(id)
          , continuation_recursion_count_(0)
        {}

        void reset()
        {
#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
            m_phase = 0;
#endif
#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
            delete_tss_storage(m_thread_data);
#else
            m_thread_data = 0;
#endif
#if defined(HPX_HAVE_APEX)
            m_apex_data = nullptr;
#endif
            m_thread_id.reset();
        }

#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
        std::size_t phase() const
        {
            return m_phase;
        }
#endif

        thread_id_type get_thread_id() const
        {
            return m_thread_id;
        }

        /*
         * Returns true if the context is runnable.
         */
        bool is_ready() const
        {
            return m_state == ctx_ready;
        }

        bool running() const
        {
            return m_state == ctx_running;
        }

        bool exited() const
        {
            return m_state == ctx_exited;
        }

        // Resume coroutine.
        // Pre:  The coroutine must be ready.
        // Post: The coroutine relinquished control. It might be ready
        //       or exited.
        // Throws:- 'abnormal_exit' if the coroutine was exited by another
        //          uncaught exception.
        // Note, it guarantees that the coroutine is resumed. Can throw only
        // on return.
        void invoke()
        {
            HPX_ASSERT(is_ready());
            do_invoke();
            // TODO: could use a binary or here to eliminate
            // shortcut evaluation (and a branch), but maybe the compiler is
            // smart enough to do it anyway as there are no side effects.
            if (m_exit_status)
            {
                if (m_exit_status == ctx_exited_return)
                    return;
                if (m_exit_status == ctx_exited_abnormally)
                {
                    std::rethrow_exception(m_type_info);
                    //std::type_info const* tinfo = nullptr;
                    //std::swap(m_type_info, tinfo);
                    //throw abnormal_exit(tinfo ? *tinfo :
                    //      typeid(unknown_exception_tag));
                }
                else {
                    HPX_ASSERT(0 && "unknown exit status");
                }
            }
        }

        // Put coroutine in ready state and relinquish control
        // to caller until resumed again.
        // Pre:  Coroutine is running.
        //       Exit not pending.
        //       Operations not pending.
        // Post: Coroutine is running.
        // Throws: exit_exception, if exit is pending *after* it has been
        //         resumed.
        void yield()
        {
            HPX_ASSERT(m_exit_state < ctx_exit_signaled); //prevent infinite loops
            HPX_ASSERT(running());

            m_state = ctx_ready;
            do_yield();

            HPX_ASSERT(running());
        }

        // Nothrow.
        ~context_base() noexcept
        {
            HPX_ASSERT(!running());
            HPX_ASSERT(exited());
            m_thread_id.reset();
#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
            delete_tss_storage(m_thread_data);
#else
            m_thread_data = 0;
#endif
#if defined(HPX_HAVE_APEX)
            m_apex_data = nullptr;
#endif
        }

        std::size_t get_thread_data() const
        {
#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
            if (!m_thread_data)
                return 0;
            return get_tss_thread_data(m_thread_data);
#else
            return m_thread_data;
#endif
        }

        std::size_t set_thread_data(std::size_t data)
        {
#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
            return set_tss_thread_data(m_thread_data, data);
#else
            std::size_t olddata = m_thread_data;
            m_thread_data = data;
            return olddata;
#endif
        }

#if defined(HPX_HAVE_APEX)
        apex_task_wrapper get_apex_data() const
        {
            // APEX wants the ADDRESS of a location to store
            // data.  This storage could be updated asynchronously,
            // so APEX stores this address, and uses it as a way
            // to remember state for the HPX thread in-betweeen
            // calls to apex::start/stop/yield/resume().
            // APEX will change the value pointed to by the address.
            return m_apex_data;
        }
        void set_apex_data(apex_task_wrapper data)
        {
            m_apex_data = data;
        }
#endif

#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
        tss_storage* get_thread_tss_data(bool create_if_needed) const
        {
            if (!m_thread_data && create_if_needed)
                m_thread_data = create_tss_storage();
            return m_thread_data;
        }
#endif

        std::size_t& get_continuation_recursion_count()
        {
            return continuation_recursion_count_;
        }

//         static std::uint64_t get_allocation_count_all(bool reset)
//         {
//             std::uint64_t count = 0;
//             for (std::size_t i = 0; i < HPX_COROUTINE_NUM_ALL_HEAPS; ++i) {
//                 count += m_allocation_counters.get(i).load();
//                 if (reset)
//                     m_allocation_counters.get(i).store(0);
//             }
//             return count;
//         }
//         static std::uint64_t get_allocation_count(std::size_t heap_num, bool reset)
//         {
//             std::uint64_t result = m_allocation_counters.get(heap_num).load();
//
//             if (reset)
//                 m_allocation_counters.get(heap_num).store(0);
//             return result;
//         }
//
//         static std::uint64_t increment_allocation_count(std::size_t heap_num)
//         {
//             return ++m_allocation_counters.get(heap_num);
//         }

    protected:
        // global coroutine state
        enum context_state
        {
            ctx_running,  // context running.
            ctx_ready,    // context at yield point.
            ctx_exited    // context is finished.
        };

        // exit request state
        enum context_exit_state
        {
            ctx_exit_not_requested,  // exit not requested.
            ctx_exit_pending,        // exit requested.
            ctx_exit_signaled        // exit request delivered.
        };

        // exit status
        enum context_exit_status
        {
            ctx_not_exited,
            ctx_exited_return,    // process exited by return.
            ctx_exited_abnormally // process exited uncleanly.
        };

        void rebind_base(thread_id_type id)
        {
            HPX_ASSERT(exited());

            m_thread_id = id;
            m_state = ctx_ready;
            m_exit_state = ctx_exit_not_requested;
            m_exit_status = ctx_not_exited;
#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
            HPX_ASSERT(m_phase == 0);
#endif
#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
            HPX_ASSERT(m_thread_data == nullptr);
#else
            HPX_ASSERT(m_thread_data == 0);
#endif
#if defined(HPX_HAVE_APEX)
            m_apex_data = rebind_base_apex(id);
#endif
            m_type_info = std::exception_ptr();
        }

        // Nothrow.
        void do_return(context_exit_status status, std::exception_ptr && info)
            noexcept
        {
            HPX_ASSERT(status != ctx_not_exited);
            HPX_ASSERT(m_state == ctx_running);
            m_type_info = std::move(info);
            m_state = ctx_exited;
            m_exit_status = status;
            do_yield();
        }

    protected:

        // Nothrow.
        void do_yield() noexcept
        {
            swap_context(*this, m_caller, detail::yield_hint());
        }

        // Nothrow.
        void do_invoke() throw ()
        {
            HPX_ASSERT(is_ready());
#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
            ++m_phase;
#endif
            m_state = ctx_running;
            swap_context(m_caller, *this, detail::invoke_hint());
        }

        typedef default_context_impl::context_impl_base ctx_type;
        ctx_type m_caller;

        static HPX_EXPORT allocation_counters m_allocation_counters;
        context_state m_state;
        context_exit_state m_exit_state;
        context_exit_status m_exit_status;
#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
        std::size_t m_phase;
#endif
#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
        mutable detail::tss_storage* m_thread_data;
#else
        mutable std::size_t m_thread_data;
#endif
#if defined(HPX_HAVE_APEX)
        // This is a pointer that APEX will use to maintain state
        // when an HPX thread is pre-empted.
        apex_task_wrapper m_apex_data;
#endif

        // This is used to generate a meaningful exception trace.
        std::exception_ptr m_type_info;
        thread_id_type m_thread_id;

        std::size_t continuation_recursion_count_;
    };
}}}}

#endif /*HPX_RUNTIME_THREADS_COROUTINES_DETAIL_CONTEXT_BASE_HPP*/
