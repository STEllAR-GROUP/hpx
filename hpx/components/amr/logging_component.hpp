//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_LOGGING_COMPONENT_NOV_10_2008_0739PM)
#define HPX_COMPONENTS_AMR_LOGGING_COMPONENT_NOV_10_2008_0739PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/components/amr/stubs/logging_component.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    class logging_component
      : public client_base<logging_component, amr::stubs::logging_component>
    {
    private:
        typedef
            client_base<logging_component, amr::stubs::logging_component>
        base_type;

    public:
        logging_component(applier::applier& app, naming::id_type gid,
                bool freeonexit = false)
          : base_type(app, gid, freeonexit)
        {}

        ///////////////////////////////////////////////////////////////////////
        void logentry(naming::id_type const& val)
        {
            this->base_type::logentry(this->gid_, val);
        }

        void logentry_sync(threads::thread_self& self, 
            naming::id_type const& val)
        {
            this->base_type::logentry_sync(self, this->gid_, val);
        }
    };

}}}

#endif
