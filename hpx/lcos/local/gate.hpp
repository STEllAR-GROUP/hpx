//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_GATE_AUG_19_2015_0749PM)
#define HPX_LCOS_LOCAL_GATE_AUG_19_2015_0749PM

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/conditional_trigger.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/assert_owns_lock.hpp>

#include <boost/thread/locks.hpp>

#include <list>
#include <utility>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex = lcos::local::spinlock>
    struct base_gate
    {
    protected:
        typedef Mutex mutex_type;

    private:
        HPX_MOVABLE_BUT_NOT_COPYABLE(base_gate)
        typedef std::list<conditional_trigger*> condition_list_type;

    public:
        /// This constructor initializes the base_gate object
        /// from the the number of participants to synchronize the control flow
        /// with.
        base_gate()
          : generation_(0)
        {}

        base_gate(base_gate && rhs)
          : promise_(std::move(rhs.promise_)),
            generation_(rhs.generation_),
            conditions_(std::move(rhs.conditions_))
        {
            rhs.generation_ = std::size_t(-1);
        }

        base_gate& operator=(base_gate && rhs)
        {
            if (this != &rhs)
            {
                boost::lock_guard<mutex_type> l(rhs.mtx_);
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
                triggered = c->set(rc) || triggered;
                if (rc && (&ec != &throws))
                    ec = rc;
            }
            return triggered;
        }

    public:
        /// \brief get a future allowing to wait for the gate to fire
        future<void> get_future(std::size_t* generation_value = 0,
            error_code& ec = hpx::throws)
        {
            boost::lock_guard<mutex_type> l(mtx_);

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

        /// \brief Set the data which has to go into the segment \a which.
        template <typename OuterLock>
        void set(OuterLock & outer_lock, error_code& ec = throws)
        {
            boost::unique_lock<mutex_type> l(mtx_);

            promise<void> p;
            std::swap(p, promise_);

            // Unlock the lock to avoid locking problems
            // when triggering the promise
            l.unlock();
            outer_lock.unlock();
            p.set_value();              // fire event

            if (&ec != &throws)
                ec = make_success_code();
        }

    protected:
        bool test_condition(std::size_t generation_value)
        {
            return !(generation_value > generation_);
        }

        struct manage_condition
        {
            manage_condition(base_gate& gate, conditional_trigger& cond)
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

            base_gate& this_;
            condition_list_type::iterator it_;
        };

    public:
        /// \brief Wait for the generational counter to reach the requested
        ///        stage.
        void synchronize(std::size_t generation_value,
            char const* function_name = "base_gate<>::synchronize",
            error_code& ec= throws)
        {
            boost::unique_lock<mutex_type> l(mtx_);
            synchronize(generation_value, l, function_name, ec);
        }

    protected:
        template <typename Lock>
        void synchronize(std::size_t generation_value, Lock& l,
            char const* function_name = "base_gate<>::synchronize",
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
                    &base_gate::test_condition, this, generation_value));

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
            boost::lock_guard<mutex_type> l(mtx_);
            HPX_ASSERT(generation_ != std::size_t(-1));
            std::size_t retval = ++generation_;

            trigger_conditions();   // re-check/trigger condition, if needed

            return retval;
        }

        std::size_t generation() const
        {
            boost::lock_guard<mutex_type> l(mtx_);
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
    struct gate : public base_gate<no_mutex>
    {
    private:
        HPX_MOVABLE_BUT_NOT_COPYABLE(gate)
        typedef base_gate<no_mutex> base_type;

    public:
        gate() {}

        template <typename Lock>
        void synchronize(std::size_t generation_value, Lock& l,
            char const* function_name = "gate::synchronize",
            error_code& ec= throws)
        {
            this->base_type::synchronize(generation_value, l, function_name, ec);
        }
    };
}}}

#endif

