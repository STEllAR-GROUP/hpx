//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_FUTURE_WAIT_AUG_02_2011_1146AM)
#define HPX_LCOS_ASYNC_FUTURE_WAIT_AUG_02_2011_1146AM

#include <hpx/hpx_fwd.hpp>

#include <vector>

#include <boost/dynamic_bitset.hpp>
#include <boost/function.hpp>

namespace hpx { namespace threads
{
    bool suspend()
    {
        // let the thread manager do other things while waiting
        threads::thread_self& self = threads::get_self();
        threads::thread_state_ex_enum statex = self.yield(threads::pending);

        if (statex == threads::wait_abort) {
            threads::thread_id_type id = self.get_thread_id();
            hpx::util::osstream strm;
            strm << "thread(" << id << ", " 
                  << threads::get_thread_description(id)
                  << ") aborted (yield returned wait_abort)";
            HPX_THROW_EXCEPTION(no_success, "threads::suspend",
                hpx::util::osstream_get_string(strm));
            return false;
        }

        return true;
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    /// The one argument version is special in the sense that it returns the 
    /// expected value directly (without wrapping it into a tuple).
    template <typename T1, typename TR1, typename F>
    inline std::size_t
    wait (lcos::future_value<T1, TR1> const& f1, F f)
    {
        f(0, f1.get());
        return 1;
    }

    // This overload of wait() will make sure that the passed function will be 
    // invoked as soon as a value gets available, it will not wait for all 
    // results to be there.
    template <typename T1, typename TR1, typename F>
    inline std::size_t
    wait (std::vector<lcos::future_value<T1, TR1> > const& lazy_values, F f)
    {
        boost::dynamic_bitset<> handled(lazy_values.size());
        while (handled.count() < lazy_values.size()) {

            bool suspended = false;
            for (std::size_t i = 0; i < lazy_values.size(); ++i) {

                // loop over all lazy_values, executing the next as soon as its
                // value gets available 
                if (!handled[i] && lazy_values[i].ready()) {
                    handled[i] = true;

                    // get the value from the future, invoke the function
                    f(i, lazy_values[i].get());

                    // give thread-manager a chance to look for more work while 
                    // waiting
                    if (!threads::suspend())
                        return handled.count();

                    suspended = true;
                }
            }

            // suspend after one full loop over all values
            if (!suspended && !threads::suspend())
                return handled.count();
        }
        return handled.count();
    }
}}

#endif
