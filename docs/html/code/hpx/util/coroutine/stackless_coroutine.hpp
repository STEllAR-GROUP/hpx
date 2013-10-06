//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COROUTINE_STACKLESS_COROUTINE_HPP_20130724
#define HPX_COROUTINE_STACKLESS_COROUTINE_HPP_20130724

#include <hpx/util/coroutine/coroutine.hpp>
#include <hpx/util/coroutine/detail/coroutine_traits.hpp>

#include <boost/move/move.hpp>
#include <boost/mpl/bool.hpp>

#include <cstddef>

namespace hpx { namespace util { namespace coroutines
{
    template <typename Signature>
    class stackless_coroutine;

    template <typename Signature>
    struct is_coroutine<stackless_coroutine<Signature> >
      : boost::mpl::true_
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Signature>
    class stackless_coroutine
    {
    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(stackless_coroutine);

        enum context_state
        {
            ctx_running,  // context running.
            ctx_ready,    // context at yield point.
            ctx_waiting,  // context waiting for events.
            ctx_exited    // context is finished.
        };

        bool running() const
        {
            return state_ == ctx_running;
        }

        bool exited() const
        {
            return state_ == ctx_exited;
        }

        typedef void (stackless_coroutine::*bool_type)();

    public:
        typedef stackless_coroutine<Signature> type;
        typedef Signature signature_type;
        typedef detail::coroutine_traits<signature_type> traits_type;

        friend struct detail::coroutine_accessor;

        typedef typename traits_type::result_type result_type;
        typedef typename traits_type::result_slot_type result_slot_type;
        typedef typename traits_type::yield_result_type yield_result_type;
        typedef typename traits_type::result_slot_traits result_slot_traits;
        typedef typename traits_type::arg_slot_type arg_slot_type;
        typedef typename traits_type::arg_slot_traits arg_slot_traits;

        typedef void* thread_id_type;

        stackless_coroutine() {}

        template <typename Functor>
        stackless_coroutine(BOOST_FWD_REF(Functor) f, thread_id_type id = 0)
          : f_(boost::forward<Functor>(f))
          , state_(ctx_ready)
          , id_(id)
#if HPX_THREAD_MAINTAIN_PHASE_INFORMATION
          , phase_(0)
#endif
#if HPX_THREAD_MAINTAIN_THREAD_DATA
          ,thread_data_(0)
#endif
        {}

        stackless_coroutine(BOOST_RV_REF(stackless_coroutine) rhs)
          : f_(boost::move(rhs.f_))
          , state_(rhs.state_)
          , id_(rhs.id_)
#if HPX_THREAD_MAINTAIN_PHASE_INFORMATION
          , phase_(rhs.phase_)
#endif
#if HPX_THREAD_MAINTAIN_THREAD_DATA
          , thread_data_(rhs.thread_data_)
#endif
        {
            rhs.id_ = 0;
            rhs.state_ = ctx_ready;
#if HPX_THREAD_MAINTAIN_PHASE_INFORMATION
            rhs.phase_ = 0;
#endif
#if HPX_THREAD_MAINTAIN_THREAD_DATA
            rhs.thread_data_ = 0;
#endif
        }

        stackless_coroutine& operator=(BOOST_RV_REF(stackless_coroutine) rhs)
        {
            stackless_coroutine(rhs).swap(*this);
            return *this;
        }

        stackless_coroutine& swap(stackless_coroutine& rhs)
        {
            std::swap(f_, rhs.f_);
            std::swap(state_, rhs.state_);
            std::swap(id_, rhs.id_);
#if HPX_THREAD_MAINTAIN_PHASE_INFORMATION
            std::swap(phase_, rhs.phase_);
#endif
#if HPX_THREAD_MAINTAIN_THREAD_DATA
            std::swap(thread_data_, rhs.thread_data_);
#endif
            return *this;
        }

        friend void swap(stackless_coroutine& lhs, stackless_coroutine& rhs)
        {
            lhs.swap(rhs);
        }

        thread_id_type get_thread_id() const
        {
            return id_;
        }

#if HPX_THREAD_MAINTAIN_PHASE_INFORMATION
        std::size_t get_thread_phase() const
        {
            return phase_;
        }
#endif

#if HPX_THREAD_MAINTAIN_THREAD_DATA
        std::size_t get_thread_data() const
        {
            return thread_data_;
        }
        std::size_t set_thread_data(std::size_t data)
        {
            std::size_t t = thread_data_;
            thread_data_ = data;
            return t;
        }
#endif

        template <typename Functor>
        void rebind(BOOST_FWD_REF(Functor) f, thread_id_type id = 0)
        {
            BOOST_ASSERT(exited());

            f_ = boost::forward<Functor>(f);
            id_ = id;
#if HPX_THREAD_MAINTAIN_PHASE_INFORMATION
            phase_ = 0;
#endif
        }

        void reset()
        {
            BOOST_ASSERT(exited());
            f_.reset();
        }

        typedef typename arg_slot_traits::template at<0>::type arg0_type;
        static const int arity = arg_slot_traits::length;

        struct yield_traits
        {
            typedef typename result_slot_traits::template at<0>::type arg0_type;
            static const int arity = result_slot_traits::length;
        };

    private:
        struct reset_on_exit
        {
            reset_on_exit(stackless_coroutine& this__)
              : this_(this__)
            {
                this_.state_ = stackless_coroutine::ctx_running;
            }

            reset_on_exit()
            {
                this_.state_ = stackless_coroutine::ctx_exited;
            }

            stackless_coroutine& this_;
        };
        friend struct reset_on_exit;

    public:
        BOOST_FORCEINLINE result_type operator()(arg0_type arg0 = arg0_type())
        {
            reset_on_exit on_exit = reset_on_exit(*this);
            HPX_UNUSED(on_exit);

            result_type result = f_(arg0);   // invoke wrapped function

            // we always have to run to completion
            BOOST_ASSERT(result == 5);       // threads::terminated == 5
            return result;
        }

        operator bool_type() const
        {
            return good() ? &stackless_coroutine::bool_type_f : 0;
        }

        bool operator==(const stackless_coroutine& rhs) const
        {
            return id_ == rhs.id_;
        }

    protected:
        void bool_type_f() {}

        bool good() const
        {
            return !exited();
        }

        hpx::util::function_nonser<signature_type> f_;
        context_state state_;
        thread_id_type id_;

#if HPX_THREAD_MAINTAIN_PHASE_INFORMATION
        std::size_t phase_;
#endif
#if HPX_THREAD_MAINTAIN_THREAD_DATA
        std::size_t thread_data_;
#endif
    };
}}}
#endif
