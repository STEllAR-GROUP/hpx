//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/component_action.hpp>
#include <hpx/async_distributed/lcos_fwd.hpp>
#include <hpx/async_distributed/transfer_continuation_action.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/get_lva.hpp>
#include <hpx/components_base/server/managed_component_base.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/parcelset/coalescing_message_handler_registration.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>

namespace hpx { namespace lcos {

    /// The \a base_lco class is the common base class for all LCO's
    /// implementing a simple set_event action
    class base_lco
    {
    public:
        virtual void set_event() = 0;

        virtual void set_exception(std::exception_ptr const& e);

        // noop by default
        virtual void connect(hpx::id_type const&);

        // noop by default
        virtual void disconnect(hpx::id_type const&);

        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this
        // component
        typedef components::managed_component<base_lco> wrapping_type;
        typedef base_lco base_type_holder;

        static components::component_type get_component_type() noexcept;
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
        void set_exception_nonvirt(std::exception_ptr const& e);

        /// The \a function connect_nonvirt is called whenever a
        /// \a connect_action is applied on a instance of a LCO. This function
        /// just forwards to the virtual function \a connect, which is
        /// overloaded by the derived concrete LCO.
        ///
        /// \param id [in] target id
        void connect_nonvirt(hpx::id_type const& id);

        /// The \a function disconnect_nonvirt is called whenever a
        /// \a disconnect_action is applied on a instance of a LCO. This function
        /// just forwards to the virtual function \a disconnect, which is
        /// overloaded by the derived concrete LCO.
        ///
        /// \param id [in] target id
        void disconnect_nonvirt(hpx::id_type const& id);

    public:
        /// Each of the exposed functions needs to be encapsulated into an action
        /// type, allowing to generate all required boilerplate code for threads,
        /// serialization, etc.
        ///
        /// The \a set_event_action may be used to unconditionally trigger any
        /// LCO instances, it carries no additional parameters.
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(
            base_lco, set_event_nonvirt, set_event_action)

        /// The \a set_exception_action may be used to transfer arbitrary error
        /// information from the remote site to the LCO instance specified as
        /// a continuation. This action carries 2 parameters:
        ///
        /// \param std::exception_ptr
        ///               [in] The exception encapsulating the error to report
        ///               to this LCO instance.
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(
            base_lco, set_exception_nonvirt, set_exception_action)

        /// The \a connect_action may be used to
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(
            base_lco, connect_nonvirt, connect_action)

        /// The \a set_exception_action may be used to
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(
            base_lco, disconnect_nonvirt, disconnect_action)
    };
}}    // namespace hpx::lcos

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    template <>
    struct get_lva<lcos::base_lco>
    {
        constexpr static lcos::base_lco* call(naming::address_type lva) noexcept
        {
            using wrapping_type = typename lcos::base_lco::wrapping_type;
            return static_cast<wrapping_type*>(lva)->get();
        }
    };

    template <>
    struct get_lva<lcos::base_lco const>
    {
        constexpr static lcos::base_lco const* call(
            naming::address_type lva) noexcept
        {
            using wrapping_type =
                std::add_const_t<typename lcos::base_lco::wrapping_type>;
            return static_cast<wrapping_type*>(lva)->get();
        }
    };
}    // namespace hpx

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

///////////////////////////////////////////////////////////////////////////////
HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DECLARATION(
    hpx::lcos::base_lco::set_event_action, "lco_set_value_action",
    std::size_t(-1), std::size_t(-1))
HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DECLARATION(
    hpx::lcos::base_lco::set_exception_action, "lco_set_value_action",
    std::size_t(-1), std::size_t(-1))
