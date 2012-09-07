//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_AND_GATE_JUL_13_2012_0919AM)
#define HPX_LCOS_LOCAL_AND_GATE_JUL_13_2012_0919AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/function.hpp>

#include <boost/dynamic_bitset.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/move/move.hpp>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    struct and_gate
    {
    private:
        typedef lcos::local::spinlock mutex_type;

    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(and_gate)

    public:
        /// \brief This constructor initializes the and_gate object from the
        ///        the number of participants to synchronize the control flow
        ///        with.
        and_gate(std::size_t count = 0)
          : received_segments_(count)
        {
        }

        and_gate(BOOST_RV_REF(and_gate) rhs)
          : received_segments_(boost::move(rhs.received_segments_)),
            promise_(boost::move(rhs.promise_))
        {
        }

        and_gate& operator=(BOOST_RV_REF(and_gate) rhs)
        {
            if (this != &rhs)
            {
                received_segments_ = boost::move(rhs.received_segments_);
                promise_ = boost::move(rhs.promise_);
            }
            return *this;
        }

        /// \brief re-initialize the gate with a different number of inputs
        void init(std::size_t count, error_code& ec = throws)
        {
            mutex_type::scoped_lock l(mtx_);

            if (0 != received_segments_.count())
            {
                // reset happens while part of the slots are filled
                HPX_THROWS_IF(ec, bad_parameter, "and_gate::init",
                    "initializing this and_gate while slots are filled");
                return;
            }

            received_segments_.resize(count);   // resize the bitmap
            received_segments_.reset();         // reset all existing bits
            promise_ = promise<void>();         // renew promise

            if (&ec != &throws)
                ec = make_success_code();
        }

        /// \brief Set the data which has to go into the segment \a which.
        void set(std::size_t which, error_code& ec = throws)
        {
            mutex_type::scoped_lock l(mtx_);
            if (which >= received_segments_.size())
            {
                // out of bounds index
                HPX_THROWS_IF(ec, bad_parameter, "and_gate::set",
                    "index is out of range for this and_gate");
                return;
            }
            if (received_segments_.test(which))
            {
                // segment already filled, logic error
                HPX_THROWS_IF(ec, bad_parameter, "and_gate::set",
                    "input with the given index has already been triggered");
                return;
            }

            // set the corresponding bit
            received_segments_.set(which);

            if (received_segments_.count() == received_segments_.size())
            {
                // we have received the last missing segment
                promise_.set_value();           // fire event
                received_segments_.reset();     // reset data store
                promise_ = promise<void>();     // renew promise
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

        /// \brief get a future allowing to wait for the gate to fire
        future<void> get_future(error_code& ec = throws)
        {
            // create new promise and return the future
            mutex_type::scoped_lock l(mtx_);
            return promise_.get_future(ec);
        }

    private:
        mutable mutex_type mtx_;
        boost::dynamic_bitset<> received_segments_;
        lcos::local::promise<void> promise_;
    };
}}}

#endif

