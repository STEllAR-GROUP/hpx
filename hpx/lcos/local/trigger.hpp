//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_TRIGGER_SEP_09_2012_1229PM)
#define HPX_LCOS_LOCAL_TRIGGER_SEP_09_2012_1229PM

#include <hpx/config.hpp>
#include <hpx/error_code.hpp>
#include <hpx/lcos/local/conditional_trigger.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/lcos/local/promise.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/assert_owns_lock.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <list>
#include <mutex>
#include <utility>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex = lcos::local::spinlock >
    struct base_trigger
    {
    protected:
        typedef Mutex mutex_type;

    private:
        HPX_MOVABLE_ONLY(base_trigger);
        typedef std::list<conditional_trigger*> condition_list_type;

    public:
        base_trigger()
          : generation_(0)
        {
        }

        base_trigger(base_trigger && rhs)
          : promise_(std::move(rhs.promise_)),
            generation_(rhs.generation_),
            conditions_(std::move(rhs.conditions_))
        {
            rhs.generation_ = std::size_t(-1);
        }

        base_trigger& operator=(base_trigger && rhs)
        {
            if (this != &rhs)
            {
                std::lock_guard<mutex_type> l(rhs.mtx_);
                promise_ = std::move(rhs.promise_);
                generation_ = rhs.generation_;
                rhs.generation_ = std::size_t(-1);
                conditions_ = std::move(rhs.conditions_);
            }
            return *this;
        }

    protected:
        bool trigger_conditions(error_code& ec = throws)
        {
            bool triggered = false;
            error_code rc(lightweight);
            for (conditional_trigger* c : conditions_)
            {
                triggered |= c->set(rc);
                if (rc && (&ec != &throws))
                    ec = rc;
            }
            return triggered;
        }

    public:
        /// \brief get a future allowing to wait for the trigger to fire
        future<void> get_future(std::size_t* generation_value = 0,
            error_code& ec = hpx::throws)
        {
            std::lock_guard<mutex_type> l(mtx_);

            HPX_ASSERT(generation_ != std::size_t(-1));
            ++generation_;

            trigger_conditions(ec);   // re-check/trigger condition, if needed
            if (!ec) {
                if (generation_value)
                    *generation_value = generation_;
                return promise_.get_future(ec);
            }
            return hpx::future<void>();
        }

        /// \brief Trigger this object.
        bool set(error_code& ec = throws)
        {
            std::lock_guard<mutex_type> l(mtx_);

            if (&ec != &throws)
                ec = make_success_code();

            promise_.set_value();           // fire event
            promise_ = promise<void>();

            if (&ec != &throws)
                ec = make_success_code();

            return true;
        }

    private:
        bool test_condition(std::size_t generation_value)
        {
            return !(generation_value > generation_);
        }

        struct manage_condition
        {
            manage_condition(base_trigger& gate, conditional_trigger& cond)
              : this_(gate)
            {
                this_.conditions_.push_back(&cond);
                it_ = this_.conditions_.end();
                --it_;      // refer to the newly added element
            }

            ~manage_condition()
            {
                this_.conditions_.erase(it_);
            }

            template <typename Condition>
            future<void> get_future(Condition&& func,
                error_code& ec = hpx::throws)
            {
                return (*it_)->get_future(std::forward<Condition>(func), ec);
            }

            base_trigger& this_;
            condition_list_type::iterator it_;
        };

    public:
        /// \brief Wait for the generational counter to reach the requested
        ///        stage.
        void synchronize(std::size_t generation_value,
            char const* function_name = "base_and_gate<>::synchronize",
            error_code& ec= throws)
        {
            std::unique_lock<mutex_type> l(mtx_);
            synchronize(generation_value, l, function_name, ec);
        }

    protected:
        template <typename Lock>
        void synchronize(std::size_t generation_value, Lock& l,
            char const* function_name = "base_and_gate<>::synchronize",
            error_code& ec= throws)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            if (generation_value < generation_)
            {
                HPX_THROWS_IF(ec, hpx::invalid_status, function_name,
                    "sequencing error, generational counter too small");
                return;
            }

           // make sure this set operation has not arrived ahead of time
            if (!test_condition(generation_value))
            {
                conditional_trigger c;
                manage_condition cond(*this, c);

                future<void> f = cond.get_future(util::bind(
                        &base_trigger::test_condition, this, generation_value));

                {
                    hpx::util::unlock_guard<Lock> ul(l);
                    f.get();
                }   // make sure lock gets re-acquired
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

    public:
        std::size_t next_generation()
        {
            std::lock_guard<mutex_type> l(mtx_);
            HPX_ASSERT(generation_ != std::size_t(-1));
            std::size_t retval = ++generation_;

            trigger_conditions();   // re-check/trigger condition, if needed

            return retval;
        }

        std::size_t generation() const
        {
            std::lock_guard<mutex_type> l(mtx_);
            return generation_;
        }

    private:
        mutable mutex_type mtx_;
        lcos::local::promise<void> promise_;
        std::size_t generation_;
        condition_list_type conditions_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Note: This type is not thread-safe. It has to be protected from
    //       concurrent access by different threads by the code using instances
    //       of this type.
    struct trigger : public base_trigger<no_mutex>
    {
    private:
        HPX_MOVABLE_ONLY(trigger);
        typedef base_trigger<no_mutex> base_type;

    public:
        trigger()
        {
        }

        trigger(trigger && rhs)
          : base_type(std::move(static_cast<base_type&>(rhs)))
        {
        }

        trigger& operator=(trigger && rhs)
        {
            if (this != &rhs)
                static_cast<base_type&>(*this) = std::move(rhs);
            return *this;
        }

        template <typename Lock>
        void synchronize(std::size_t generation_value, Lock& l,
            char const* function_name = "trigger::synchronize",
            error_code& ec= throws)
        {
            this->base_type::synchronize(generation_value, l, function_name, ec);
        }
    };
}}}

#endif
