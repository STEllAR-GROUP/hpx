//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_MEMORY_JUN_25_2008_0122PM)
#define HPX_COMPONENTS_MEMORY_JUN_25_2008_0122PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class memory
    {
    public:
        typedef memory type_holder;

        // parcel action code: the action to be performed on the destination
        // object
        enum actions
        {
            memory_store8 = 0,        ///< store a 8 bit value to a memory location
            memory_store16 = 1,       ///< store a 16 bit value to a memory location
            memory_store32 = 2,       ///< store a 32 bit value to a memory location
            memory_store64 = 3,       ///< store a 64 bit value to a memory location
            memory_load8 = 4,         ///< load a 8 bit value from a memory location
            memory_load16 = 5,        ///< load a 16 bit value from a memory location
            memory_load32 = 6,        ///< load a 32 bit value from a memory location
            memory_load64 = 7         ///< load a 64 bit value from a memory location
        };

        static component_type get_component_type()
        {
            return components::get_component_type<memory>();
        }
        static void set_component_type(component_type t)
        {
            components::set_component_type<memory>(t);
        }

        // constructor
        memory()
        {}

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        ///
        /// \param self [in] The PX \a thread used to execute this function.
        /// \param appl [in] The applier to be used for finalization of the
        ///             component instance.
        void finalize() {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// \brief Action to store an 8 bit value to a memory location
        void store8(boost::uint64_t addr, boost::uint8_t value)
        {
            *reinterpret_cast<boost::uint8_t*>(addr) = value;
        }

        /// \brief Action to store an 16 bit value to a memory location
        void store16(boost::uint64_t addr, boost::uint16_t value)
        {
            *reinterpret_cast<boost::uint16_t*>(addr) = value;
        }

        /// \brief Action to store an 32 bit value to a memory location
        void store32(boost::uint64_t addr, boost::uint32_t value)
        {
            *reinterpret_cast<boost::uint32_t*>(addr) = value;
        }

        /// \brief Action to store an 64 bit value to a memory location
        void store64(boost::uint64_t addr, boost::uint64_t value)
        {
            *reinterpret_cast<boost::uint64_t*>(addr) = value;
        }

        /// \brief Action to load an 8 bit value to a memory location
        boost::uint8_t load8(boost::uint64_t addr)
        {
            return *reinterpret_cast<boost::uint8_t*>(addr);
        }

        /// \brief Action to load an 16 bit value to a memory location
        boost::uint16_t load16(boost::uint64_t addr)
        {
            return *reinterpret_cast<boost::uint16_t*>(addr);
        }

        /// \brief Action to load an 32 bit value to a memory location
        boost::uint32_t load32(boost::uint64_t addr)
        {
            return *reinterpret_cast<boost::uint32_t*>(addr);
        }

        /// \brief Action to load an 64 bit value to a memory location
        boost::uint64_t load64(boost::uint64_t addr)
        {
            return *reinterpret_cast<boost::uint64_t*>(addr);
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::direct_action2<
            memory, memory_store8, boost::uint64_t, boost::uint8_t,
            &memory::store8
        > store8_action;

        typedef hpx::actions::direct_action2<
            memory, memory_store16, boost::uint64_t, boost::uint16_t,
            &memory::store16
        > store16_action;

        typedef hpx::actions::direct_action2<
            memory, memory_store32, boost::uint64_t, boost::uint32_t,
            &memory::store32
        > store32_action;

        typedef hpx::actions::direct_action2<
            memory, memory_store64, boost::uint64_t, boost::uint64_t,
            &memory::store64
        > store64_action;

        typedef hpx::actions::direct_result_action1<
            memory, boost::uint8_t, memory_load8, boost::uint64_t,
            &memory::load8
        > load8_action;

        typedef hpx::actions::direct_result_action1<
            memory, boost::uint16_t, memory_load16, boost::uint64_t,
            &memory::load16
        > load16_action;

        typedef hpx::actions::direct_result_action1<
            memory, boost::uint32_t, memory_load32, boost::uint64_t,
            &memory::load32
        > load32_action;

        typedef hpx::actions::direct_result_action1<
            memory, boost::uint64_t, memory_load64, boost::uint64_t,
            &memory::load64
        > load64_action;

    };
}}}

///////////////////////////////////////////////////////////////////////////////
// Declaration of serialization support for the runtime_support actions
HPX_REGISTER_ACTION_DECLARATION_EX(hpx::components::server::memory::store8_action, store8_action)
HPX_REGISTER_ACTION_DECLARATION_EX(hpx::components::server::memory::store16_action, store16_action)
HPX_REGISTER_ACTION_DECLARATION_EX(hpx::components::server::memory::store32_action, store32_action)
HPX_REGISTER_ACTION_DECLARATION_EX(hpx::components::server::memory::store64_action, store64_action)

HPX_REGISTER_ACTION_DECLARATION_EX(hpx::components::server::memory::load8_action, load8_action)
HPX_REGISTER_ACTION_DECLARATION_EX(hpx::components::server::memory::load16_action, load16_action)
HPX_REGISTER_ACTION_DECLARATION_EX(hpx::components::server::memory::load32_action, load32_action)
HPX_REGISTER_ACTION_DECLARATION_EX(hpx::components::server::memory::load64_action, load64_action)

///////////////////////////////////////////////////////////////////////////////
// make sure all needed action::get_action_name() functions get defined
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<boost::uint8_t>::set_value_action,
    set_value_action_uint8_t)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<boost::uint16_t>::set_value_action,
    set_value_action_uint16_t)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<boost::uint32_t>::set_value_action,
    set_value_action_uint32_t)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<boost::uint64_t>::set_value_action,
    set_value_action_uint64_t)

#endif

