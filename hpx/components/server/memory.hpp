//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_MEMORY_JUN_25_2008_0121PM)
#define HPX_COMPONENTS_MEMORY_JUN_25_2008_0121PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/components/action.hpp>

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
            memory_store32 = 0,       ///< store a 32 bit value to a memory location
            memory_store64 = 1,       ///< store a 64 bit value to a memory location
        };

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = component_memory };

        // constructor
        memory() 
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// \brief Action to store a value to a memory location
        threadmanager::thread_state store32(
            threadmanager::px_thread_self& self, applier::applier& app,
            boost::uint64_t addr, boost::uint32_t value)
        {
            BOOST_ASSERT(false);    // should never be called
            return threadmanager::terminated;
        }

        threadmanager::thread_state store64(
            threadmanager::px_thread_self& self, applier::applier& app,
            boost::uint64_t addr, boost::uint64_t value)
        {
            BOOST_ASSERT(false);    // should never be called
            return threadmanager::terminated;
        }

        ///
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

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef direct_action2<
            memory, memory_store32, boost::uint64_t, boost::uint32_t, 
            &memory::store32, &memory::local_store32
        > store32_action;

        typedef direct_action2<
            memory, memory_store64, boost::uint64_t, boost::uint64_t, 
            &memory::store64, &memory::local_store64
        > store64_action;
    };

}}}

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the runtime_support actions
HPX_SERIALIZE_ACTION(hpx::components::server::memory::store32_action);
HPX_SERIALIZE_ACTION(hpx::components::server::memory::store64_action);

#endif

