//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(SERVER_STARTUP_SHUTDOWN_NOV_23_2011_0706PM)
#define SERVER_STARTUP_SHUTDOWN_NOV_23_2011_0706PM

namespace startup_shutdown { namespace server
{
    class HPX_COMPONENT_EXPORT startup_shutdown_component
      : public hpx::components::simple_component_base<startup_shutdown_component>
    {
    public:
        // parcel action code: the action to be performed on the destination
        // object (the accumulator)
        enum actions
        {
            startup_shutdown_component_init = 0,
        };

        // constructor: initialize accumulator value
        startup_shutdown_component()
          : arg_(0)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        void init(std::string const& option)
        {
            arg_ = option;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action1<
            startup_shutdown_component, startup_shutdown_component_init,
            std::string const&, &startup_shutdown_component::init
        > init_action;

    private:
      std::string arg_;
    };
}}

HPX_REGISTER_ACTION_DECLARATION(
    startup_shutdown::server::startup_shutdown_component::init_action,
    startup_shutdown_component_init_action);

#endif
