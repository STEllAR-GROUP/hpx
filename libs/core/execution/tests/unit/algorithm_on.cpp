//  Copyright (c) 2023 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/type_support/meta.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

template <class S>
struct scheduler_env
{
    template <typename CPO>
    friend S tag_invoke(
        ex::get_completion_scheduler_t<CPO>, const scheduler_env&) noexcept
    {
        return {};
    }
};

//! Scheduler that executes everything inline, i.e., on the same thread
struct inline_scheduler
{
    template <typename R>
    struct operation
    {
        R recv_;

        friend void tag_invoke(ex::start_t, operation& self) noexcept
        {
            ex::set_value((R &&) self.recv_);
        }
    };

    struct my_sender
    {
        using is_sender = void;
        using completion_signatures =
            ex::completion_signatures<ex::set_value_t()>;

        template <typename R>
        friend operation<R> tag_invoke(ex::connect_t, my_sender self, R&& r)
        {
            return {{}, (R &&) r};
        }

        friend scheduler_env<inline_scheduler> tag_invoke(
            ex::get_env_t, const my_sender&) noexcept
        {
            return {};
        }
    };

    friend my_sender tag_invoke(ex::schedule_t, inline_scheduler)
    {
        return {};
    }

    friend bool operator==(inline_scheduler, inline_scheduler) noexcept
    {
        return true;
    }

    friend bool operator!=(inline_scheduler, inline_scheduler) noexcept
    {
        return false;
    }
};

int main()
{
    {
        auto snd = ex::on(inline_scheduler{}, ex::just(13));
        static_assert(ex::is_sender_v<decltype(snd)>);
        (void) snd;
    }

    return hpx::util::report_errors();
}
