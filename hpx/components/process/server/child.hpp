// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PROCESS_SERVER_CHILD_MAR_11_2016_0646PM)
#define HPX_PROCESS_SERVER_CHILD_MAR_11_2016_0646PM

#include <hpx/config.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <type_traits>

#include <boost/cstdint.hpp>

#include <hpx/components/process/util/child.hpp>
#include <hpx/components/process/util/execute.hpp>
#include <hpx/components/process/export_definitions.hpp>

namespace hpx { namespace components { namespace process { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_PROCESS_EXPORT child : public component_base<child>
    {
    public:
        template <typename ... Ts>
        child(Ts && ... ts)
          : child_(process::util::execute(std::forward<Ts>(ts)...))
        {}

        void terminate();
        boost::uint32_t wait_for_exit();

        HPX_DEFINE_COMPONENT_ACTION(child, terminate);
        HPX_DEFINE_COMPONENT_ACTION(child, wait_for_exit);

    private:
        process::util::child child_;
    };
}}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::process::server::child::terminate_action,
    hpx_components_process_server_child_terminate_action);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::process::server::child::wait_for_exit_action,
    hpx_components_process_server_child_wait_for_exit);

#endif

