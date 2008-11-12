//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STUBS_LOGGING_COMPONENT_NOV_10_2008_0739PM)
#define HPX_COMPONENTS_AMR_STUBS_LOGGING_COMPONENT_NOV_10_2008_0739PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/components/amr/server/logging_component.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace stubs 
{
    ///////////////////////////////////////////////////////////////////////////
    class logging_component
      : public components::stubs::stub_base<amr::server::logging_component>
    {
    private:
        typedef
            components::stubs::stub_base<amr::server::logging_component> 
        base_type;

    public:
        logging_component(applier::applier& appl)
          : base_type(appl)
        {}

        ///////////////////////////////////////////////////////////////////////
        static void logentry(applier::applier& appl, 
            naming::id_type const& gid, naming::id_type const& val)
        {
            typedef amr::server::logging_component::logentry_action action_type;
            appl.apply<action_type>(gid, val);
        }

        void logentry(naming::id_type const& gid, naming::id_type const& val)
        {
            logentry(this->appl_, gid, val);
        }

        static void logentry_sync(threads::thread_self& self, 
            applier::applier& appl, naming::id_type const& gid, 
            naming::id_type const& val)
        {
            typedef amr::server::logging_component::logentry_action action_type;
            lcos::eager_future<action_type, void>(appl, gid, val).get(self);
        }

        void logentry_sync(threads::thread_self& self, 
            naming::id_type const& gid, naming::id_type const& val)
        {
            logentry_sync(self, this->appl_, gid, val);
        }
    };

}}}}

#endif
