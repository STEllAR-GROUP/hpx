//  Copyright (c) 2016 Bibek Wagle
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_POOL_TIMER)
#define HPX_UTIL_POOL_TIMER

#include <hpx/config.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/steady_clock.hpp>

#include <boost/system/error_code.hpp>

#include <memory>
#include <string>

namespace hpx { namespace util { namespace detail
{
    class pool_timer;
}}}

namespace hpx { namespace util
{
    class HPX_EXPORT pool_timer
    {
        HPX_NON_COPYABLE(pool_timer);

    public:
        pool_timer();

        pool_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            std::string const& description = "",
            bool pre_shutdown = true);

        ~pool_timer();

        bool start(util::steady_duration const& time_duration,
            bool evaluate = false);
        bool stop();

        bool is_started() const;
        bool is_terminated() const;

    private:
        std::shared_ptr<detail::pool_timer> timer_;
    };
}}


#endif
