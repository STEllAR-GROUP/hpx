//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_BASE_LCO_JUN_12_2008_0852PM)
#define HPX_LCOS_BASE_LCO_JUN_12_2008_0852PM

#include <hpx/config.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/ini.hpp>

namespace hpx { namespace lcos
{
    /// The \a base_lco class is the common base class for all LCO's
    /// implementing a simple set_event action
    class HPX_API_EXPORT base_lco
    {
    public:
        virtual void set_event () = 0;

        virtual void set_exception (boost::exception_ptr const& e);

        // noop by default
        virtual void connect(naming::id_type const &);

        // noop by default
        virtual void disconnect(naming::id_type const &);

        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this
        // component
        typedef components::managed_component<base_lco> wrapping_type;
        typedef base_lco base_type_holder;

        static components::component_type get_component_type();
        static void set_component_type(components::component_type type);

        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco();

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        virtual void finalize();

        /// The \a function set_event_nonvirt is called whenever a
        /// \a set_event_action is applied on a instance of a LCO. This function
        /// just forwards to the virtual function \a set_event, which is
        /// overloaded by the derived concrete LCO.
        void set_event_nonvirt();

        /// The \a function set_exception is called whenever a
        /// \a set_exception_action is applied on a instance of a LCO. This function
        /// just forwards to the virtual function \a set_exception, which is
        /// overloaded by the derived concrete LCO.
        ///
        /// \param e      [in] The exception encapsulating the error to report
        ///               to this LCO instance.
        void set_exception_nonvirt (boost::exception_ptr const& e);

        /// The \a function connect_nonvirt is called whenever a
        /// \a connect_action is applied on a instance of a LCO. This function
        /// just forwards to the virtual function \a connect, which is
        /// overloaded by the derived concrete LCO.
        ///
        /// \param id [in] target id
        void connect_nonvirt(naming::id_type const & id);

        /// The \a function disconnect_nonvirt is called whenever a
        /// \a disconnect_action is applied on a instance of a LCO. This function
        /// just forwards to the virtual function \a disconnect, which is
        /// overloaded by the derived concrete LCO.
        ///
        /// \param id [in] target id
        void disconnect_nonvirt(naming::id_type const & id);

    public:
        /// Each of the exposed functions needs to be encapsulated into an action
        /// type, allowing to generate all required boilerplate code for threads,
        /// serialization, etc.
        ///
        /// The \a set_event_action may be used to unconditionally trigger any
        /// LCO instances, it carries no additional parameters.
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(base_lco, set_event_nonvirt,
            set_event_action);

        /// The \a set_exception_action may be used to transfer arbitrary error
        /// information from the remote site to the LCO instance specified as
        /// a continuation. This action carries 2 parameters:
        ///
        /// \param boost::exception_ptr
        ///               [in] The exception encapsulating the error to report
        ///               to this LCO instance.
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(base_lco, set_exception_nonvirt,
            set_exception_action);

        /// The \a connect_action may be used to
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(base_lco, connect_nonvirt,
            connect_action);

        /// The \a set_exception_action may be used to
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(base_lco, disconnect_nonvirt,
            disconnect_action);
    };
}}

///////////////////////////////////////////////////////////////////////////////
// Declaration of serialization support for the base LCO actions
HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco::set_event_action, base_set_event_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco::set_exception_action, base_set_exception_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco::connect_action, base_connect_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco::disconnect_action, base_disconnect_action)

#endif
