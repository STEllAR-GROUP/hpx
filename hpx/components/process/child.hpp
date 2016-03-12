// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PROCESS_CHILD_MAR_11_2016_0646PM)
#define HPX_PROCESS_CHILD_MAR_11_2016_0646PM

#include <hpx/config.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>

#include <hpx/components/process/server/child.hpp>

#include <boost/cstdint.hpp>

namespace hpx { namespace components { namespace process
{
    ///////////////////////////////////////////////////////////////////////////
    class child : public client_base<child, process::server::child>
    {
        typedef client_base<child, process::server::child> base_type;

    public:
        template <typename ... Ts>
        child(Ts &&... ts)
          : base_type(std::forward<Ts>(ts)...)
        {}

        hpx::future<void> terminate()
        {
            typedef server::child::terminate_action terminate_action;
            return hpx::async(terminate_action(), this->get_id());
        }

        hpx::future<boost::uint32_t> wait_for_exit()
        {
            typedef server::child::wait_for_exit_action wait_for_exit_action;
            return hpx::async(wait_for_exit_action(), this->get_id());
        }
    };
}}}

#endif
