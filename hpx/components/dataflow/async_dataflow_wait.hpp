//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_DATAFLOW_WAIT_HPP)
#define HPX_LCOS_ASYNC_DATAFLOW_WAIT_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/dataflow/dataflow.hpp>

#include <vector>

#include <boost/dynamic_bitset.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    // This overload of wait() will make sure that the passed function will be
    // invoked as soon as a value becomes available, it will not wait for all
    // results to be there.
    template <typename T1, typename TR1, typename F>
    inline std::size_t
    wait (std::vector<lcos::dataflow_base<T1, TR1> > const& dataflows, F const& f,
        boost::int64_t suspend_for = 10)
    {
        boost::dynamic_bitset<> handled(dataflows.size());
        std::size_t handled_count = 0;

        std::vector<lcos::future<T1, TR1> > lazy_values;
        lazy_values.reserve(dataflows.size());
        typedef dataflow_base<T1, TR1> dataflow_type;
        BOOST_FOREACH(dataflow_type const & d, dataflows)
        {
            lazy_values.push_back(d.get_future());
        }

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
                    this_thread::suspend();
                    suspended = true;
                }
            }

            // suspend after one full loop over all values, 10ms should be fine
            // (default parameter)
            if (!suspended)
                this_thread::suspend(boost::posix_time::milliseconds(suspend_for));
        }
        return handled.count();
    }

    template <typename F>
    inline std::size_t
    wait (std::vector<lcos::dataflow_base<void> > const& dataflows, F const& f,
        std::size_t suspend_for = 10)
    {
        boost::dynamic_bitset<> handled(dataflows.size());
        std::size_t handled_count = 0;

        std::vector<lcos::future<void> > lazy_values;
        lazy_values.reserve(dataflows.size());
        BOOST_FOREACH(dataflow_base<void> const & d, dataflows)
        {
            lazy_values.push_back(d.get_future());
        }

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
                    this_thread::suspend();
                    suspended = true;
                }
            }

            // suspend after one full loop over all values, 10ms should be fine
            // (default parameter)
            if (!suspended)
                this_thread::suspend(boost::posix_time::milliseconds(suspend_for));
        }
        return handled.count();
    }
}}

#endif
