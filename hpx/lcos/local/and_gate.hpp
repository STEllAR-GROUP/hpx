//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_AND_GATE_JUL_13_2012_0919AM)
#define HPX_LCOS_LOCAL_AND_GATE_JUL_13_2012_0919AM

#include <hpx/config.hpp>
#include <hpx/error_code.hpp>
#include <hpx/lcos/local/conditional_trigger.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/assert_owns_lock.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <boost/dynamic_bitset.hpp>

#include <list>
#include <mutex>
#include <utility>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex = lcos::local::spinlock>
    struct base_and_gate
    {
    protected:
        typedef Mutex mutex_type;

    private:
        HPX_MOVABLE_ONLY(base_and_gate);
        typedef std::list<conditional_trigger*> condition_list_type;

    public:
        /// \brief This constructor initializes the base_and_gate object from the
        ///        the number of participants to synchronize the control flow
        ///        with.
        base_and_gate(std::size_t count = 0)
          : received_segments_(count), generation_(0)
        {
        }

        base_and_gate(base_and_gate && rhs)
          : received_segments_(std::move(rhs.received_segments_)),
            promise_(std::move(rhs.promise_)),
            generation_(rhs.generation_),
            conditions_(std::move(rhs.conditions_))
        {
            rhs.generation_ = std::size_t(-1);
        }

        base_and_gate& operator=(base_and_gate && rhs)
        {
            if (this != &rhs)
            {
                std::lock_guard<mutex_type> l(rhs.mtx_);
                received_segments_ = std::move(rhs.received_segments_);
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
            if (!conditions_.empty())
            {
                error_code rc(lightweight);
                for (conditional_trigger* c : conditions_)
                {
                    triggered |= c->set(rc);
                    if (rc && (&ec != &throws))
                        ec = rc;
                }
            }
            else
            {
                if (&ec != &throws)
                    ec = make_success_code();
            }
            return triggered;
        }

    public:
        /// \brief get a future allowing to wait for the gate to fire
        future<void> get_future(std::size_t count = std::size_t(-1),
            std::size_t* generation_value = nullptr,
            error_code& ec = hpx::throws)
        {
            std::unique_lock<mutex_type> l(mtx_);

            // by default we use as many segments as specified during construction
            if (count == std::size_t(-1))
                count = received_segments_.size();
            HPX_ASSERT(count != 0);

            init_locked(l, count, ec);
            if (!ec) {
                HPX_ASSERT(generation_ != std::size_t(-1));
                ++generation_;

                trigger_conditions(ec);   // re-check/trigger condition, if needed
                if (!ec) {
                    if (generation_value)
                        *generation_value = generation_;
                    return promise_.get_future(ec);
                }
            }
            return hpx::future<void>();
        }

        /// \brief Set the data which has to go into the segment \a which.
        template <typename OuterLock>
        bool set(std::size_t which, OuterLock & outer_lock, error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(mtx_);
            if (which >= received_segments_.size())
            {
                // out of bounds index
                l.unlock();
                HPX_THROWS_IF(ec, bad_parameter, "base_and_gate<>::set",
                    "index is out of range for this base_and_gate");
                return false;
            }
            if (received_segments_.test(which))
            {
                // segment already filled, logic error
                l.unlock();
                HPX_THROWS_IF(ec, bad_parameter, "base_and_gate<>::set",
                    "input with the given index has already been triggered");
                return false;
            }

            if (&ec != &throws)
                ec = make_success_code();

            // set the corresponding bit
            received_segments_.set(which);

            if (received_segments_.count() == received_segments_.size())
            {
                // we have received the last missing segment
                promise<void> p;
                std::swap(p, promise_);
                received_segments_.reset();     // reset data store
                {
                    // Unlock the lock to avoid locking problems
                    // when triggering the promise
                    l.unlock();
                    outer_lock.unlock();
                    p.set_value();              // fire event
                    return true;
                }
            }

            return false;
        }

        bool set(std::size_t which, error_code& ec = throws)
        {
            no_mutex mtx;
            std::unique_lock<no_mutex> lk(mtx);
            return set(which, lk, ec);
        }

    protected:
        bool test_condition(std::size_t generation_value)
        {
            return !(generation_value > generation_);
        }

        struct manage_condition
        {
            manage_condition(base_and_gate& gate, conditional_trigger& cond)
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

            base_and_gate& this_;
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
                l.unlock();
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
                    &base_and_gate::test_condition, this, generation_value));

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

    protected:
        template <typename Lock>
        void init_locked(Lock& l, std::size_t count, error_code& ec = throws)
        {
            if (0 != received_segments_.count())
            {
                // reset happens while part of the slots are filled
                l.unlock();
                HPX_THROWS_IF(ec, bad_parameter, "base_and_gate<>::init",
                    "initializing this base_and_gate while slots are filled");
                return;
            }

            if (received_segments_.size() != count)
                received_segments_.resize(count);   // resize the bitmap
            received_segments_.reset();         // reset all existing bits

            if (&ec != &throws)
                ec = make_success_code();
        }

    private:
        mutable mutex_type mtx_;
        boost::dynamic_bitset<> received_segments_;
        lcos::local::promise<void> promise_;
        std::size_t generation_;
        condition_list_type conditions_;
    };


    ///////////////////////////////////////////////////////////////////////////
    // Note: This type is not thread-safe. It has to be protected from
    //       concurrent access by different threads by the code using instances
    //       of this type.
    struct and_gate : public base_and_gate<no_mutex>
    {
    private:
        HPX_MOVABLE_ONLY(and_gate);
        typedef base_and_gate<no_mutex> base_type;

    public:
        and_gate(std::size_t count = 0)
          : base_type(count)
        {
        }

        and_gate(and_gate && rhs)
          : base_type(std::move(static_cast<base_type&>(rhs)))
        {
        }

        and_gate& operator=(and_gate && rhs)
        {
            if (this != &rhs)
                static_cast<base_type&>(*this) = std::move(rhs);
            return *this;
        }

        template <typename Lock>
        void synchronize(std::size_t generation_value, Lock& l,
            char const* function_name = "and_gate::synchronize",
            error_code& ec= throws)
        {
            this->base_type::synchronize(generation_value, l, function_name, ec);
        }
    };
}}}

#endif

