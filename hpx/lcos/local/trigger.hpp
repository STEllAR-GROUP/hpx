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

    public:
        trigger()
          : generation_(0)
        {
        }

        trigger(BOOST_RV_REF(trigger) rhs)
          : promise_(boost::move(rhs.promise_)),
            generation_(rhs.generation_),
            condition_(boost::move(rhs.condition_))
        {
            rhs.generation_ = std::size_t(-1);
        }

        trigger& operator=(BOOST_RV_REF(trigger) rhs)
        {
            if (this != &rhs)
            {
                promise_ = boost::move(rhs.promise_);
                generation_ = rhs.generation_;
                rhs.generation_ = std::size_t(-1);
                condition_ = boost::move(rhs.condition_);
            }
            return *this;
        }

        /// \brief get a future allowing to wait for the trigger to fire
        future<void> get_future(error_code& ec = hpx::throws)
        {
            mutex_type::scoped_lock l(mtx_);

            BOOST_ASSERT(generation_ != std::size_t(-1));
            ++generation_;

            condition_.set();     // re-check/trigger condition, if needed

            return promise_.get_future(ec);
        }

        /// \brief Trigger this object.
        bool set(error_code& ec = throws)
        {
            mutex_type::scoped_lock l(mtx_);

            if (promise_.is_ready())
            {
                // segment already filled, logic error
                HPX_THROWS_IF(ec, bad_parameter, "trigger::set",
                    "input with the given index has already been triggered");
                return false;
            }

            // trigger this object
            condition_.reset();
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

    public:
        /// \brief Wait for the generational counter to reach the requested
        ///        stage.
        void synchronize(std::size_t generation,
            char const* function_name = "trigger::synchronize",
            error_code& ec= throws)
        {
            mutex_type::scoped_lock l(mtx_);

            if (generation < generation_)
            {
                HPX_THROWS_IF(ec, hpx::invalid_status, function_name,
                    "sequencing error, generational counter too small");
                return;
            }

           // make sure this set operation has not arrived ahead of time
            if (!test_condition(generation))
            {
                future<void> f = condition_.get_future(util::bind(
                        &trigger::test_condition, this, generation));

                hpx::util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                f.get();
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

        std::size_t next_generation()
        {
            mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(generation_ != std::size_t(-1));
            std::size_t retval = ++generation_;

            condition_.set();     // re-check/trigger condition, if needed

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
        conditional_trigger condition_;
    };
}}}

#endif
