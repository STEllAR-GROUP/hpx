//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_TRIGGER_SEP_09_2012_1229PM)
#define HPX_LCOS_LOCAL_TRIGGER_SEP_09_2012_1229PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/conditional_trigger.hpp>
#include <hpx/lcos/local/no_mutex.hpp>

#include <boost/move/move.hpp>
#include <boost/assert.hpp>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex = lcos::local::spinlock >
    struct base_trigger
    {
    protected:
        typedef Mutex mutex_type;

    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(base_trigger)
        typedef std::list<conditional_trigger*> condition_list_type;

    public:
        base_trigger()
          : generation_(0)
        {
        }

        base_trigger(BOOST_RV_REF(base_trigger) rhs)
          : promise_(boost::move(rhs.promise_)),
            generation_(rhs.generation_),
            conditions_(boost::move(rhs.conditions_))
        {
            rhs.generation_ = std::size_t(-1);
        }

        base_trigger& operator=(BOOST_RV_REF(base_trigger) rhs)
        {
            if (this != &rhs)
            {
                typename mutex_type::scoped_lock l(rhs.mtx_);
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
            typename mutex_type::scoped_lock l(mtx_);

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
            typename mutex_type::scoped_lock l(mtx_);

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

    private:
        bool test_condition(std::size_t generation)
        {
            return !(generation > generation_);
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

            future<void> get_future(HPX_STD_FUNCTION<bool()> const& func,
                error_code& ec = hpx::throws)
            {
                return (*it_)->get_future(func, ec);
            }

            base_trigger& this_;
            condition_list_type::iterator it_;
        };

    public:
        /// \brief Wait for the generational counter to reach the requested
        ///        stage.
        void synchronize(std::size_t generation,
            char const* function_name = "base_and_gate<>::synchronize",
            error_code& ec= throws)
        {
            typename mutex_type::scoped_lock l(mtx_);
            synchronize(generation, l, function_name, ec);
        }

    protected:
        template <typename Lock>
        void synchronize(std::size_t generation, Lock& l,
            char const* function_name = "base_and_gate<>::synchronize",
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
                        &base_trigger::test_condition, this, generation));

                {
                    hpx::util::scoped_unlock<Lock> ul(l);
                    f.get();
                }   // make sure lock gets re-acquired
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

    public:
        std::size_t next_generation()
        {
            typename mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(generation_ != std::size_t(-1));
            std::size_t retval = ++generation_;

            trigger_conditions();   // re-check/trigger condition, if needed

            return retval;
        }

        std::size_t generation() const
        {
            typename mutex_type::scoped_lock l(mtx_);
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
        BOOST_MOVABLE_BUT_NOT_COPYABLE(trigger)
        typedef base_trigger<no_mutex> base_type;

    public:
        trigger()
        {
        }

        trigger(BOOST_RV_REF(trigger) rhs)
          : base_type(boost::move(static_cast<base_type&>(rhs)))
        {
        }

        trigger& operator=(BOOST_RV_REF(trigger) rhs)
        {
            if (this != &rhs)
                static_cast<base_type&>(*this) = boost::move(rhs);
            return *this;
        }

        template <typename Lock>
        void synchronize(std::size_t generation, Lock& l,
            char const* function_name = "trigger::synchronize",
            error_code& ec= throws)
        {
            this->base_type::synchronize(generation, l, function_name, ec);
        }
    };
}}}

#endif
