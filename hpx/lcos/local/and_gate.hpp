//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_AND_GATE_JUL_13_2012_0919AM)
#define HPX_LCOS_LOCAL_AND_GATE_JUL_13_2012_0919AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/conditional_trigger.hpp>
#include <hpx/lcos/local/no_mutex.hpp>

#include <boost/dynamic_bitset.hpp>
#include <boost/move/move.hpp>
#include <boost/assert.hpp>
#include <boost/foreach.hpp>

#include <list>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex = lcos::local::spinlock>
    struct base_and_gate
    {
    protected:
        typedef Mutex mutex_type;

    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(base_and_gate)
        typedef std::list<conditional_trigger*> condition_list_type;

    public:
        /// \brief This constructor initializes the base_and_gate object from the
        ///        the number of participants to synchronize the control flow
        ///        with.
        base_and_gate(std::size_t count = 0)
          : received_segments_(count), generation_(0)
        {
        }

        base_and_gate(BOOST_RV_REF(base_and_gate) rhs)
          : received_segments_(boost::move(rhs.received_segments_)),
            promise_(boost::move(rhs.promise_)),
            generation_(rhs.generation_),
            conditions_(boost::move(rhs.conditions_))
        {
            rhs.generation_ = std::size_t(-1);
        }

        base_and_gate& operator=(BOOST_RV_REF(base_and_gate) rhs)
        {
            if (this != &rhs)
            {
                typename mutex_type::scoped_lock l(rhs.mtx_);
                received_segments_ = boost::move(rhs.received_segments_);
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
        /// \brief get a future allowing to wait for the gate to fire
        future<void> get_future(std::size_t count,
            std::size_t* generation = 0, error_code& ec = hpx::throws)
        {
            typename mutex_type::scoped_lock l(mtx_);
            init_locked(count, ec);
            if (!ec) {
                BOOST_ASSERT(generation_ != std::size_t(-1));
                ++generation_;

                trigger_conditions(ec);   // re-check/trigger condition, if needed
                if (!ec) {
                    if (generation) 
                        *generation = generation_;
                    return promise_.get_future(ec);
                }
            }
            return hpx::future<void>();
        }

        /// \brief Set the data which has to go into the segment \a which.
        bool set(std::size_t which, error_code& ec = throws)
        {
            typename mutex_type::scoped_lock l(mtx_);
            if (which >= received_segments_.size())
            {
                // out of bounds index
                HPX_THROWS_IF(ec, bad_parameter, "base_and_gate<>::set",
                    "index is out of range for this base_and_gate");
                return false;
            }
            if (received_segments_.test(which))
            {
                // segment already filled, logic error
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
                promise_.set_value();           // fire event
                promise_ = promise<void>();
                received_segments_.reset();     // reset data store
                return true;
            }

            return false;
        }

    protected:
        bool test_condition(std::size_t generation)
        {
            return !(generation > generation_);
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

            future<void> get_future(HPX_STD_FUNCTION<bool()> const& func,
                error_code& ec = hpx::throws)
            {
                return (*it_)->get_future(func, ec);
            }

            base_and_gate& this_;
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
                        &base_and_gate::test_condition, this, generation));

                {
                    hpx::util::unlock_the_lock<Lock> ul(l);
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

    protected:
        void init_locked(std::size_t count, error_code& ec = throws)
        {
            if (0 != received_segments_.count())
            {
                // reset happens while part of the slots are filled
                HPX_THROWS_IF(ec, bad_parameter, "base_and_gate<>::init",
                    "initializing this base_and_gate while slots are filled");
                return;
            }

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
        BOOST_MOVABLE_BUT_NOT_COPYABLE(and_gate)
        typedef base_and_gate<no_mutex> base_type;

    public:
        and_gate()
        {
        }

        and_gate(BOOST_RV_REF(and_gate) rhs)
          : base_type(boost::move(static_cast<base_type&>(rhs)))
        {
        }

        and_gate& operator=(BOOST_RV_REF(and_gate) rhs)
        {
            if (this != &rhs)
                static_cast<base_type&>(*this) = boost::move(rhs);
            return *this;
        }

        template <typename Lock>
        void synchronize(std::size_t generation, Lock& l,
            char const* function_name = "and_gate::synchronize",
            error_code& ec= throws)
        {
            this->base_type::synchronize(generation, l, function_name, ec);
        }
    };
}}}

#endif

