//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_MEMORY_JUN_25_2008_0122PM)
#define HPX_COMPONENTS_MEMORY_JUN_25_2008_0122PM

#include <hpx/config.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/traits/is_component.hpp>
#include <hpx/util/integer/uint128.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <type_traits>

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

        typedef util::integer::uint128 uint128_t;

        // constructor
        memory() = default;

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        HPX_CXX14_CONSTEXPR static void finalize() {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// \brief Action to store an 8 bit value to a memory location
        void store8(std::uint64_t addr, std::uint8_t value)
        {
            *reinterpret_cast<std::uint8_t*>(addr) = value;
        }

        /// \brief Action to store an 16 bit value to a memory location
        void store16(std::uint64_t addr, std::uint16_t value)
        {
            *reinterpret_cast<std::uint16_t*>(addr) = value;
        }

        /// \brief Action to store an 32 bit value to a memory location
        void store32(std::uint64_t addr, std::uint32_t value)
        {
            *reinterpret_cast<std::uint32_t*>(addr) = value;
        }

        /// \brief Action to store an 64 bit value to a memory location
        void store64(std::uint64_t addr, std::uint64_t value)
        {
            *reinterpret_cast<std::uint64_t*>(addr) = value;
        }

        /// \brief Action to store an 64 bit value to a memory location
        void store128(std::uint64_t addr, uint128_t const& value)
        {
            *reinterpret_cast<uint128_t*>(addr) = value;
        }

        /// \brief Action to load an 8 bit value to a memory location
        std::uint8_t load8(std::uint64_t addr) const
        {
            return *reinterpret_cast<std::uint8_t*>(addr);
        }

        /// \brief Action to load an 16 bit value to a memory location
        std::uint16_t load16(std::uint64_t addr) const
        {
            return *reinterpret_cast<std::uint16_t*>(addr);
        }

        /// \brief Action to load an 32 bit value to a memory location
        std::uint32_t load32(std::uint64_t addr) const
        {
            return *reinterpret_cast<std::uint32_t*>(addr);
        }

        /// \brief Action to load an 64 bit value to a memory location
        std::uint64_t load64(std::uint64_t addr) const
        {
            return *reinterpret_cast<std::uint64_t*>(addr);
        }

        /// \brief Action to load an 128 bit value to a memory location
        uint128_t load128(std::uint64_t addr) const
        {
            return *reinterpret_cast<uint128_t*>(addr);
        }

        ///////////////////////////////////////////////////////////////////////
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(memory, store8);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(memory, store16);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(memory, store32);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(memory, store64);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(memory, store128);

        HPX_DEFINE_COMPONENT_DIRECT_ACTION(memory, load8);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(memory, load16);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(memory, load32);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(memory, load64);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(memory, load128);

        // This component type requires valid id for its actions to be invoked
        HPX_CONSTEXPR static bool is_target_valid(naming::id_type const&)
        {
            return true;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT naming::gid_type allocate(std::size_t size);

    HPX_DEFINE_PLAIN_ACTION(allocate, allocate_action);
}}}

namespace hpx { namespace traits
{
    // memory is a (hand-rolled) component
    template <>
    struct is_component<components::server::memory>
      : std::true_type
    {};
}}

///////////////////////////////////////////////////////////////////////////////
// Declaration of serialization support for the runtime_support actions
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::allocate_action, allocate_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::memory::store8_action, store8_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::memory::store16_action, store16_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::memory::store32_action, store32_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::memory::store64_action, store64_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::memory::store128_action, store128_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::memory::load8_action, load8_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::memory::load16_action, load16_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::memory::load32_action, load32_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::memory::load64_action, load64_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::memory::load128_action, load128_action)

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    hpx::components::server::memory::uint128_t, hpx_components_memory_uint128_t)

#endif

