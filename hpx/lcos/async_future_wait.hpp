//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_FUTURE_WAIT_AUG_02_2011_1146AM)
#define HPX_LCOS_ASYNC_FUTURE_WAIT_AUG_02_2011_1146AM

#include <hpx/hpx_fwd.hpp>

#include <vector>

#include <boost/dynamic_bitset.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    /// The one argument version is special in the sense that it returns the
    /// expected value directly (without wrapping it into a tuple).
    template <typename T1, typename TR1, typename F>
    inline std::size_t
    wait (lcos::promise<T1, TR1> const& f1, F const& f)
    {
        f(0, f1.get());
        return 1;
    }

    template <typename F>
    inline std::size_t
    wait (lcos::promise<void> const& f1, F const& f)
    {
        f1.get();
        f(0);
        return 1;
    }

    // This overload of wait() will make sure that the passed function will be
    // invoked as soon as a value becomes available, it will not wait for all
    // results to be there.
    template <typename T1, typename TR1, typename F>
    inline std::size_t
    wait (std::vector<lcos::promise<T1, TR1> > const& lazy_values, F const& f,
        std::size_t suspend_for = 10)
    {
        boost::dynamic_bitset<> handled(lazy_values.size());
        std::size_t handled_count = 0;
        while (handled_count < lazy_values.size()) {

            bool suspended = false;
            for (std::size_t i = 0; i < lazy_values.size(); ++i) {

                // loop over all lazy_values, executing the next as soon as its
                // value becomes available
                if (!handled[i] && lazy_values[i].is_ready()) {
                    // get the value from the future, invoke the function
                    f(i, lazy_values[i].get());

                    handled[i] = true;
                    ++handled_count;

                    // give thread-manager a chance to look for more work while
                    // waiting
                    threads::suspend();
                    suspended = true;
                }
            }

            // suspend after one full loop over all values, 10ms should be fine
            // (default parameter)
            if (!suspended)
                threads::suspend(boost::posix_time::milliseconds(suspend_for));
        }
        return handled.count();
    }

    template <typename F>
    inline std::size_t
    wait (std::vector<lcos::promise<void> > const& lazy_values, F const& f,
        std::size_t suspend_for = 10)
    {
        boost::dynamic_bitset<> handled(lazy_values.size());
        std::size_t handled_count = 0;
        while (handled_count < lazy_values.size()) {

            bool suspended = false;
            for (std::size_t i = 0; i < lazy_values.size(); ++i) {

                // loop over all lazy_values, executing the next as soon as its
                // value becomes available
                if (!handled[i] && lazy_values[i].is_ready()) {
                    // get the value from the future, invoke the function
                    lazy_values[i].get();
                    f(i);

                    handled[i] = true;
                    ++handled_count;

                    // give thread-manager a chance to look for more work while
                    // waiting
                    threads::suspend();
                    suspended = true;
                }
            }

            // suspend after one full loop over all values, 10ms should be fine
            // (default parameter)
            if (!suspended)
                threads::suspend(boost::posix_time::milliseconds(suspend_for));
        }
        return handled.count();
    }
}}

#endif
