//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_LOGGING_NOV_10_2008_0719PM)
#define HPX_COMPONENTS_AMR_LOGGING_NOV_10_2008_0719PM

#include <hpx/components/amr/server/logging_component.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    /// This class implements a simple logging sink. It exposes the function
    /// \a logentry which is supposed to record the received values in a 
    /// application specific manner.
    class HPX_COMPONENT_EXPORT logging 
      : public amr::server::logging_component
    {
    private:
        typedef amr::server::logging_component base_type;

    public:
        logging(threads::thread_self& self, applier::applier& appl)
          : base_type(self, appl)
        {}

        static component_type get_component_type()
        {
            return components::get_component_type<base_type>();
        }
        static void set_component_type(component_type type)
        {
            components::set_component_type<base_type>(type);
        }

        /// This is the function implementing the logging functionality
        /// It takes the values as calculated during the current time step.
        threads::thread_state logentry(threads::thread_self&, 
            applier::applier&, naming::id_type const& memblock_gid);
    };

}}}

#endif
