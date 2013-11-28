//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_CONDITIONAL_TRIGGER_SEP_09_2012_1256PM)
#define HPX_LCOS_LOCAL_CONDITIONAL_TRIGGER_SEP_09_2012_1256PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/function.hpp>

#include <boost/move/move.hpp>
#include <hpx/util/assert.hpp>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    struct conditional_trigger
    {
    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(conditional_trigger)

    public:
        conditional_trigger()
        {
        }

        conditional_trigger(BOOST_RV_REF(conditional_trigger) rhs)
          : cond_(boost::move(rhs.cond_))
        {
        }

        conditional_trigger& operator=(BOOST_RV_REF(conditional_trigger) rhs)
        {
            if (this != &rhs)
            {
                promise_ = boost::move(rhs.promise_);
                cond_ = boost::move(rhs.cond_);
            }
            return *this;
        }

        /// \brief get a future allowing to wait for the trigger to fire
        future<void> get_future(HPX_STD_FUNCTION<bool()> const& func,
            error_code& ec = hpx::throws)
        {
            cond_ = func;

            future<void> f = promise_.get_future(ec);

            set(ec);      // trigger as soon as possible

            return f;
        }

        void reset()
        {
            cond_.clear();
        }

        /// \brief Trigger this object.
        bool set(error_code& ec = throws)
        {
            if (promise_.is_ready())
            {
                // segment already filled, logic error
                HPX_THROWS_IF(ec, bad_parameter, "conditional_trigger::set",
                    "input with the given index has already been triggered");
                return false;
            }

            if (&ec != &throws)
                ec = make_success_code();

            // trigger this object
            if (cond_ && cond_())
            {
                promise_.set_value();           // fire event
                promise_ = promise<void>();
                return true;
            }

            return false;
        }

    private:
        lcos::local::promise<void> promise_;
        HPX_STD_FUNCTION<bool()> cond_;
    };
}}}

#endif
