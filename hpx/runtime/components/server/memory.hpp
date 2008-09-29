//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_MEMORY_JUN_25_2008_0122PM)
#define HPX_COMPONENTS_MEMORY_JUN_25_2008_0122PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/actions/action.hpp>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class memory
    {
    public:
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
            memory_load64 = 7,        ///< load a 64 bit value from a memory location
        };

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        static component_type get_component_type() 
        { 
            return component_memory; 
        }

        // constructor
        memory() 
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// \brief Action to store an 8 bit value to a memory location
        threads::thread_state store8(threads::thread_self& self, 
            applier::applier& app, boost::uint64_t addr, boost::uint8_t value)
        {
            local_store8(app, addr, value);
            return threads::terminated;
        }

        /// \brief Action to store a 16 bit value to a memory location
        threads::thread_state store16(threads::thread_self& self, 
            applier::applier& app, boost::uint64_t addr, boost::uint16_t value)
        {
            local_store16(app, addr, value);
            return threads::terminated;
        }

        /// \brief Action to store a 32 value value to a memory location
        threads::thread_state store32(threads::thread_self& self, 
            applier::applier& app, boost::uint64_t addr, boost::uint32_t value)
        {
            local_store32(app, addr, value);
            return threads::terminated;
        }

        /// \brief Action to store a 64 value value to a memory location
        threads::thread_state store64(threads::thread_self& self, 
            applier::applier& app, boost::uint64_t addr, boost::uint64_t value)
        {
            local_store64(app, addr, value);
            return threads::terminated;
        }

        /// \brief Action to load an 8 bit value to a memory location
        threads::thread_state load8(threads::thread_self& self, 
            applier::applier& app, boost::uint8_t* value, boost::uint64_t addr)
        {
            *value = local_load8(app, addr);
            return threads::terminated;
        }

        /// \brief Action to load a 16 bit value to a memory location
        threads::thread_state load16(threads::thread_self& self, 
            applier::applier& app, boost::uint16_t* value, boost::uint64_t addr)
        {
            *value = local_load16(app, addr);
            return threads::terminated;
        }

        /// \brief Action to load a 32 bit value to a memory location
        threads::thread_state load32(threads::thread_self& self, 
            applier::applier& app, boost::uint32_t* value, boost::uint64_t addr)
        {
            *value = local_load32(app, addr);
            return threads::terminated;
        }

        /// \brief Action to load a 64 bit value to a memory location
        threads::thread_state load64(threads::thread_self& self, 
            applier::applier& app, boost::uint64_t* value, boost::uint64_t addr)
        {
            *value = local_load64(app, addr);
            return threads::terminated;
        }

        ///
        void local_store8(applier::applier& app, boost::uint64_t addr, 
            boost::uint8_t value)
        {
            *reinterpret_cast<boost::uint8_t*>(addr) = value;
        }

        void local_store16(applier::applier& app, boost::uint64_t addr, 
            boost::uint16_t value)
        {
            *reinterpret_cast<boost::uint16_t*>(addr) = value;
        }

        void local_store32(applier::applier& app, boost::uint64_t addr, 
            boost::uint32_t value)
        {
            *reinterpret_cast<boost::uint32_t*>(addr) = value;
        }

        void local_store64(applier::applier& app, boost::uint64_t addr, 
            boost::uint64_t value)
        {
            *reinterpret_cast<boost::uint64_t*>(addr) = value;
        }

        ///
        boost::uint8_t local_load8(applier::applier& app, boost::uint64_t addr)
        {
            return *reinterpret_cast<boost::uint8_t*>(addr);
        }

        boost::uint16_t local_load16(applier::applier& app, boost::uint64_t addr)
        {
            return *reinterpret_cast<boost::uint16_t*>(addr);
        }

        boost::uint32_t local_load32(applier::applier& app, boost::uint64_t addr)
        {
            return *reinterpret_cast<boost::uint32_t*>(addr);
        }

        boost::uint64_t local_load64(applier::applier& app, boost::uint64_t addr)
        {
            return *reinterpret_cast<boost::uint64_t*>(addr);
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::direct_action2<
            memory, memory_store8, boost::uint64_t, boost::uint8_t, 
            &memory::store8, &memory::local_store8
        > store8_action;

        typedef hpx::actions::direct_action2<
            memory, memory_store16, boost::uint64_t, boost::uint16_t, 
            &memory::store16, &memory::local_store16
        > store16_action;

        typedef hpx::actions::direct_action2<
            memory, memory_store32, boost::uint64_t, boost::uint32_t, 
            &memory::store32, &memory::local_store32
        > store32_action;

        typedef hpx::actions::direct_action2<
            memory, memory_store64, boost::uint64_t, boost::uint64_t, 
            &memory::store64, &memory::local_store64
        > store64_action;

        typedef hpx::actions::direct_result_action1<
            memory, boost::uint8_t, memory_load8, boost::uint64_t, 
            &memory::load8, &memory::local_load8
        > load8_action;

        typedef hpx::actions::direct_result_action1<
            memory, boost::uint16_t, memory_load16, boost::uint64_t, 
            &memory::load16, &memory::local_load16
        > load16_action;

        typedef hpx::actions::direct_result_action1<
            memory, boost::uint32_t, memory_load32, boost::uint64_t, 
            &memory::load32, &memory::local_load32
        > load32_action;

        typedef hpx::actions::direct_result_action1<
            memory, boost::uint64_t, memory_load64, boost::uint64_t, 
            &memory::load64, &memory::local_load64
        > load64_action;

    };

}}}

#endif

