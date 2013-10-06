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

#include <boost/move/move.hpp>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class memory
    {
    public:
        typedef memory type_holder;

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
        HPX_DEFINE_COMPONENT_ACTION(memory, store8);
        HPX_DEFINE_COMPONENT_ACTION(memory, store16);
        HPX_DEFINE_COMPONENT_ACTION(memory, store32);
        HPX_DEFINE_COMPONENT_ACTION(memory, store64);

        HPX_DEFINE_COMPONENT_ACTION(memory, load8);
        HPX_DEFINE_COMPONENT_ACTION(memory, load16);
        HPX_DEFINE_COMPONENT_ACTION(memory, load32);
        HPX_DEFINE_COMPONENT_ACTION(memory, load64);

        /// This is the default hook implementation for decorate_action which 
        /// does no hooking at all.
        static HPX_STD_FUNCTION<threads::thread_function_type> 
        wrap_action(HPX_STD_FUNCTION<threads::thread_function_type> f,
            naming::address::address_type)
        {
            return boost::move(f);
        }
    };
}}}

///////////////////////////////////////////////////////////////////////////////
// Declaration of serialization support for the runtime_support actions
HPX_REGISTER_ACTION_DECLARATION(hpx::components::server::memory::store8_action, store8_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::components::server::memory::store16_action, store16_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::components::server::memory::store32_action, store32_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::components::server::memory::store64_action, store64_action)

HPX_REGISTER_ACTION_DECLARATION(hpx::components::server::memory::load8_action, load8_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::components::server::memory::load16_action, load16_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::components::server::memory::load32_action, load32_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::components::server::memory::load64_action, load64_action)

#endif

