//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_AND_GATE_JUL_13_2012_0919AM)
#define HPX_LCOS_LOCAL_AND_GATE_JUL_13_2012_0919AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>
// #include <hpx/lcos/local/conditional_trigger.hpp>

#include <boost/dynamic_bitset.hpp>
#include <boost/move/move.hpp>
#include <boost/assert.hpp>

namespace hpx { namespace lcos { namespace local
{
    struct no_mutex
    {
        typedef boost::unique_lock<no_mutex> scoped_lock;
        typedef boost::detail::try_lock_wrapper<spinlock> scoped_try_lock;

        void lock() {}
        void unlock() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex = lcos::local::spinlock>
    struct and_gate
    {
    protected:
        typedef Mutex mutex_type;

    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(and_gate)

    public:
        /// \brief This constructor initializes the and_gate object from the
        ///        the number of participants to synchronize the control flow
        ///        with.
        and_gate(std::size_t count = 0)
          : received_segments_(count), generation_(0)
        {
        }

        and_gate(BOOST_RV_REF(and_gate) rhs)
          : received_segments_(boost::move(rhs.received_segments_)),
            promise_(boost::move(rhs.promise_)),
            generation_(rhs.generation_) //,
//             condition_(boost::move(rhs.condition_))
        {
            rhs.generation_ = std::size_t(-1);
        }

        and_gate& operator=(BOOST_RV_REF(and_gate) rhs)
        {
            if (this != &rhs)
            {
                typename mutex_type::scoped_lock l(rhs.mtx_);
                received_segments_ = boost::move(rhs.received_segments_);
                promise_ = boost::move(rhs.promise_);
                generation_ = rhs.generation_;
                rhs.generation_ = std::size_t(-1);
//                 condition_ = boost::move(rhs.condition_);
            }
            return *this;
        }

//         /// \brief re-initialize the gate with a different number of inputs
//         std::size_t init(std::size_t count, error_code& ec = hpx::throws)
//         {
//             typename mutex_type::scoped_lock l(this->mtx_);
//             init_locked(count, ec);
//             if (ec)
//                 return std::size_t(-1);
//
//             BOOST_ASSERT(generation_ != std::size_t(-1));
//             return ++generation_;
//         }

        /// \brief get a future allowing to wait for the gate to fire
        future<void> get_future(std::size_t count,
            std::size_t* generation = 0, error_code& ec = hpx::throws)
        {
            typename mutex_type::scoped_lock l(mtx_);
            init_locked(count, ec);
            if (!ec) {
                BOOST_ASSERT(generation_ != std::size_t(-1));
                ++generation_;

//                 condition_.set(ec);   // re-check/trigger condition, if needed
//                 if (!ec)
                if (generation)
                    *generation = generation_;
                return promise_.get_future(ec);
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
                HPX_THROWS_IF(ec, bad_parameter, "and_gate::set",
                    "index is out of range for this and_gate");
                return false;
            }
            if (received_segments_.test(which))
            {
                // segment already filled, logic error
                HPX_THROWS_IF(ec, bad_parameter, "and_gate::set",
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
//                 condition_.reset();
                promise_.set_value();           // fire event
                promise_ = promise<void>();
                received_segments_.reset();     // reset data store
                return true;
            }

            return false;
        }

//     protected:
//         bool test_condition(std::size_t generation)
//         {
//             return !(generation > generation_);
//         }

    public:
        /// \brief Wait for the generational counter to reach the requested
        ///        stage.
        void synchronize(std::size_t generation,
            char const* function_name = "and_gate::synchronize",
            error_code& ec= throws)
        {
            typename mutex_type::scoped_lock l(mtx_);
            synchronize(generation, l, function_name, ec);
        }

        template <typename Lock>
        void synchronize(std::size_t generation, Lock& l,
            char const* function_name = "and_gate::synchronize",
            error_code& ec= throws)
        {
            if (generation < generation_)
            {
                HPX_THROWS_IF(ec, hpx::invalid_status, function_name,
                    "sequencing error, generational counter too small");
                return;
            }

           // make sure this set operation has not arrived ahead of time
//             if (!test_condition(generation))
//             {
//                 future<void> f = condition_.get_future(util::bind(
//                         &and_gate::test_condition, this, generation));
//
//                 hpx::util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
//                 f.get();
//             }

            while (generation > generation_)
            {
                hpx::util::unlock_the_lock<Lock> ul(l);
                hpx::this_thread::suspend(hpx::threads::pending, function_name);
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

        std::size_t next_generation()
        {
            typename mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(generation_ != std::size_t(-1));
            std::size_t retval = ++generation_;

//             condition_.set();     // re-check/trigger condition, if needed

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
                HPX_THROWS_IF(ec, bad_parameter, "and_gate::init",
                    "initializing this and_gate while slots are filled");
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
//         conditional_trigger condition_;
    };
}}}

#endif

