//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_TRIGGER_SEP_09_2012_1229PM)
#define HPX_LCOS_LOCAL_TRIGGER_SEP_09_2012_1229PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/conditional_trigger.hpp>

#include <boost/move/move.hpp>
#include <boost/assert.hpp>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    struct trigger
    {
    protected:
        typedef lcos::local::spinlock mutex_type;

    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(trigger)
        typedef std::list<conditional_trigger*> condition_list_type;

    public:
        trigger()
          : generation_(0)
        {
        }

        trigger(BOOST_RV_REF(trigger) rhs)
          : promise_(boost::move(rhs.promise_)),
            generation_(rhs.generation_),
            conditions_(boost::move(rhs.conditions_))
        {
            rhs.generation_ = std::size_t(-1);
        }

        trigger& operator=(BOOST_RV_REF(trigger) rhs)
        {
            if (this != &rhs)
            {
                mutex_type::scoped_lock l(rhs.mtx_);
                promise_ = boost::move(rhs.promise_);
                generation_ = rhs.generation_;
                rhs.generation_ = std::size_t(-1);
                conditions_ = boost::move(rhs.conditions_);
            }
            return *this;
        }

    protected:
        bool trigger_conditions(error_code& ec = throws)
        {
            bool triggered = false;
            error_code rc(lightweight);
            BOOST_FOREACH(conditional_trigger* c, conditions_)
            {
                triggered |= c->set(rc);
                if (rc && (&ec != &throws)) 
                    ec = rc;
            }
            return triggered;
        }

    public:
        /// \brief get a future allowing to wait for the trigger to fire
        future<void> get_future(std::size_t* generation = 0, 
            error_code& ec = hpx::throws)
        {
            mutex_type::scoped_lock l(mtx_);

            BOOST_ASSERT(generation_ != std::size_t(-1));
            ++generation_;

            trigger_conditions(ec);   // re-check/trigger condition, if needed
            if (!ec) {
                if (generation) 
                    *generation = generation_;
                return promise_.get_future(ec);
            }
            return hpx::future<void>();
        }

        /// \brief Trigger this object.
        bool set(error_code& ec = throws)
        {
            mutex_type::scoped_lock l(mtx_);

            if (promise_.is_ready())
            {
                // segment already filled, logic error
                HPX_THROWS_IF(ec, bad_parameter, "trigger::set",
                    "input has already been triggered");
                return false;
            }

            if (&ec != &throws)
                ec = make_success_code();

            promise_.set_value();           // fire event
            promise_ = promise<void>();

            if (&ec != &throws)
                ec = make_success_code();

            return true;
        }

    protected:
        bool test_condition(std::size_t generation)
        {
            return !(generation > generation_);
        }

        struct manage_condition
        {
            manage_condition(trigger& gate, conditional_trigger& cond)
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

            future<void> get_future(HPX_STD_FUNCTION<bool()> const& func,
                error_code& ec = hpx::throws)
            {
                return (*it_)->get_future(func, ec);
            }

            trigger& this_;
            condition_list_type::iterator it_;
        };

    public:
        /// \brief Wait for the generational counter to reach the requested
        ///        stage.
//         void synchronize(std::size_t generation,
//             char const* function_name = "and_gate::synchronize",
//             error_code& ec= throws)
//         {
//             mutex_type::scoped_lock l(mtx_);
//             synchronize(generation, l, function_name, ec);
//         }

        template <typename Lock>
        void synchronize(std::size_t generation, Lock& l,
            char const* function_name = "and_gate::synchronize",
            error_code& ec= throws)
        {
            BOOST_ASSERT(l.owns_lock());

            if (generation < generation_)
            {
                HPX_THROWS_IF(ec, hpx::invalid_status, function_name,
                    "sequencing error, generational counter too small");
                return;
            }

           // make sure this set operation has not arrived ahead of time
            if (!test_condition(generation))
            {
                conditional_trigger c;
                manage_condition cond(*this, c);

                future<void> f = cond.get_future(util::bind(
                        &trigger::test_condition, this, generation));

                {
                    hpx::util::unlock_the_lock<Lock> ul(l);
                    f.get();
                }   // make sure lock gets re-acquired
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

        std::size_t next_generation()
        {
            mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(generation_ != std::size_t(-1));
            std::size_t retval = ++generation_;

            trigger_conditions();   // re-check/trigger condition, if needed

            return retval;
        }

        std::size_t generation() const
        {
            mutex_type::scoped_lock l(mtx_);
            return generation_;
        }

    private:
        mutable mutex_type mtx_;
        lcos::local::promise<void> promise_;
        std::size_t generation_;
        condition_list_type conditions_;
    };
}}}

#endif
