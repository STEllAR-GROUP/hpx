//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/futures.hpp>

#include <utility>

namespace hpx::lcos::local {

    ///////////////////////////////////////////////////////////////////////////
    struct conditional_trigger
    {
    public:
        conditional_trigger() = default;

        conditional_trigger(conditional_trigger const& rhs) = delete;
        conditional_trigger& operator=(conditional_trigger const& rhs) = delete;

        conditional_trigger(conditional_trigger&& rhs) = default;
        conditional_trigger& operator=(conditional_trigger&& rhs) = default;

        /// Get a future allowing to wait for the trigger to fire
        template <typename Condition>
        hpx::future<void> get_future(
            Condition&& func, error_code& ec = hpx::throws)
        {
            cond_.assign(HPX_FORWARD(Condition, func));

            hpx::future<void> f = promise_.get_future(ec);

            set(ec);    // trigger as soon as possible

            return f;
        }

        void reset()
        {
            cond_.reset();
        }

        /// Trigger this object.
        bool set(error_code& ec = throws)
        {
            if (&ec != &throws)
                ec = make_success_code();

            // trigger this object
            if (cond_ && cond_())
            {
                promise_.set_value();    // fire event
                promise_ = hpx::promise<void>();
                return true;
            }

            return false;
        }

    private:
        hpx::promise<void> promise_;
        hpx::function<bool()> cond_;
    };
}    // namespace hpx::lcos::local
