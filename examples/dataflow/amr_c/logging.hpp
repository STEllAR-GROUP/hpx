//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_LOGGING_NOV_10_2008_0719PM)
#define HPX_COMPONENTS_AMR_LOGGING_NOV_10_2008_0719PM

#include <hpx/lcos/local/mutex.hpp>
#include "stencil_data.hpp"
#include "../parameter.hpp"
#include <hpx/lcos/barrier.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server
{
    /// This class implements a simple logging sink. It exposes the function
    /// \a logentry which is supposed to record the received values in a
    /// application specific manner.
    class HPX_COMPONENT_EXPORT logging
      : public simple_component_base<logging>
    {
    private:
        typedef simple_component_base<logging> base_type;

    public:
        logging() { count = 0;}

        /// This is the function implementing the logging functionality
        /// It takes the values as calculated during the current time step.
        void logentry(stencil_data const& memblock_gid, int row, int column, parameter const& par );

        /// Each of the exposed functions needs to be encapsulated into an action
        /// type, allowing to generate all required boilerplate code for threads,
        /// serialization, etc.
        ///
        /// The \a set_value_action may be used to trigger any LCO instances
        /// while carrying an additional parameter of any type.
        ///
        /// \param Result [in] The type of the result to be transferred back to
        ///               this LCO instance.
        HPX_DEFINE_COMPONENT_ACTION(logging, logentry);

    private:
        typedef lcos::local::mutex mutex_type;
        static mutex_type mtx_;
        std::size_t count;
    };

}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct logging : public components::stub_base<amr::server::logging>
    {
        ///////////////////////////////////////////////////////////////////////
        static void logentry(naming::id_type const& gid,
            stencil_data const& val, int row, int column, parameter const& par)
        {
            typedef amr::server::logging::logentry_action action_type;
            hpx::apply<action_type>(gid, val, row, column,par);
        }
    };

}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr
{
    ///////////////////////////////////////////////////////////////////////////
    class logging : public client_base<logging, amr::stubs::logging>
    {
    private:
        typedef client_base<logging, amr::stubs::logging> base_type;

    public:
        logging() {}

        logging(naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        void logentry(stencil_data const& val, int row,int column, parameter const& par)
        {
            this->base_type::logentry(this->get_gid(), val, row, column, par);
        }
    };

}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::amr::server::logging::logentry_action
  , dataflow_logentry_action
);

#endif
