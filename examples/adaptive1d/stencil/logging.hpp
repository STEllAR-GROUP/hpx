//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STENCIL_LOGGING_AUG_02_2011_0719PM)
#define HPX_COMPONENTS_STENCIL_LOGGING_AUG_02_2011_0719PM

#include <hpx/lcos/local/mutex.hpp>
#include "stencil_data.hpp"
#include "../parameter.hpp"
#include <hpx/lcos/barrier.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace adaptive1d { namespace server
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

        enum actions
        {
            logging_logentry = 0,
        };

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
        typedef hpx::actions::action4<
            logging, logging_logentry, stencil_data const&, int,int,parameter const&,
            &logging::logentry
        > logentry_action;

    private:
        typedef lcos::local::mutex mutex_type;
        static mutex_type mtx_;
        std::size_t count;
    };

}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace adaptive1d { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct logging : public components::stub_base<adaptive1d::server::logging>
    {
        ///////////////////////////////////////////////////////////////////////
        static void logentry(naming::id_type const& gid,
            stencil_data const& val, int row, int column, parameter const& par)
        {
            typedef adaptive1d::server::logging::logentry_action action_type;
            hpx::apply<action_type>(gid, val, row, column,par);
        }
    };

}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace adaptive1d
{
    ///////////////////////////////////////////////////////////////////////////
    class logging : public client_base<logging, adaptive1d::stubs::logging>
    {
    private:
        typedef client_base<logging, adaptive1d::stubs::logging> base_type;

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
    hpx::components::adaptive1d::server::logging::logentry_action
  , adaptive1d_logentry_action
);

#endif
