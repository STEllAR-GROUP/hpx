//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_CONDITIONAL_TRIGGER_SEP_09_2012_1256PM)
#define HPX_LCOS_LOCAL_CONDITIONAL_TRIGGER_SEP_09_2012_1256PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/promise.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/function.hpp>

#include <utility>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    struct conditional_trigger
    {
    private:
        HPX_MOVABLE_ONLY(conditional_trigger);

    public:
        conditional_trigger()
        {
        }

        conditional_trigger(conditional_trigger && rhs)
          : cond_(std::move(rhs.cond_))
        {
        }

        conditional_trigger& operator=(conditional_trigger && rhs)
        {
            if (this != &rhs)
            {
                promise_ = std::move(rhs.promise_);
                cond_ = std::move(rhs.cond_);
            }
            return *this;
        }

        /// \brief get a future allowing to wait for the trigger to fire
        template <typename Condition>
        future<void> get_future(Condition&& func,
            error_code& ec = hpx::throws)
        {
            cond_.assign(std::forward<Condition>(func));

            future<void> f = promise_.get_future(ec);

            set(ec);      // trigger as soon as possible

            return f;
        }

        void reset()
        {
            cond_.reset();
        }

        /// \brief Trigger this object.
        bool set(error_code& ec = throws)
        {
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
        util::function_nonser<bool()> cond_;
    };
}}}

#endif
