//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/applier/applier.hpp>
#include <hpx/async_distributed/applier/bind_naming_wrappers.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/traits/is_component.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming/credit_handling.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/components/server/create_component_fwd.hpp>
#include <hpx/runtime/components_fwd.hpp>
#include <hpx/runtime_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <type_traits>
#include <vector>

namespace hpx { namespace components
{

template <typename Component>
class fixed_component;

///////////////////////////////////////////////////////////////////////////
template <typename Component>
class fixed_component_base : public traits::detail::fixed_component_tag
{
private:
    typedef typename std::conditional<
            std::is_same<Component, detail::this_type>::value,
            fixed_component_base, Component
        >::type this_component_type;

    Component& derived()
    {
        return static_cast<Component&>(*this);
    }
    Component const& derived() const
    {
        return static_cast<Component const&>(*this);
    }

public:
    typedef this_component_type wrapped_type;
    typedef this_component_type base_type_holder;
    typedef fixed_component<this_component_type> wrapping_type;

    /// \brief Construct an empty fixed_component
    fixed_component_base(std::uint64_t msb, std::uint64_t lsb)
      : msb_(msb)
      , lsb_(lsb)
    {}

    ~fixed_component_base() = default;

    /// \brief finalize() will be called just before the instance gets
    ///        destructed
    void finalize()
    {
        /// Unbind the GID if it's not this instantiations fixed gid and is
        /// is not invalid.
        if (naming::invalid_gid != gid_)
        {
            error_code ec(lightweight);     // ignore errors
            applier::unbind_gid_local(gid_, ec);
            gid_ = naming::gid_type();      // invalidate GID
        }
    }

private:
    // declare friends which are allowed to access get_base_gid()
    template <typename Component_, typename...Ts>
    friend naming::gid_type server::create(Ts&&... ts);

    template <typename Component_, typename...Ts>
    friend naming::gid_type server::create_migrated(
        naming::gid_type const& gid, void** p, Ts&&...ts);

    template <typename Component_, typename...Ts>
    friend std::vector<naming::gid_type> bulk_create(std::size_t count, Ts&&...ts);

    // Return the component's fixed GID.
    naming::gid_type get_base_gid(
        naming::gid_type const& assign_gid = naming::invalid_gid) const
    {
        if (assign_gid)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "fixed_component_base::get_base_gid",
                "fixed_components must be assigned new gids on creation");
            return naming::invalid_gid;
        }

        if (!gid_)
        {
            naming::address addr(get_locality(),
                components::get_component_type<wrapped_type>(),
                std::size_t(static_cast<this_component_type const*>(this)));

            gid_ = naming::gid_type(msb_, lsb_);

            // Try to bind the preset GID first
            if (!applier::bind_gid_local(gid_, addr))
            {
                std::ostringstream strm;
                strm << "could not bind_gid(local): " << gid_;
                gid_ = naming::gid_type();   // invalidate GID
                HPX_THROW_EXCEPTION(duplicate_component_address,
                    "fixed_component_base<Component>::get_base_gid",
                    strm.str());
            }
        }
        return gid_;
    }

public:
    naming::id_type get_id() const
    {
        // fixed_address components are created without any credits
        naming::gid_type gid = derived().get_base_gid();
        HPX_ASSERT(!naming::detail::has_credits(gid));

        naming::detail::replenish_credits(gid);
        return naming::id_type(gid, naming::id_type::managed);
    }

    naming::id_type get_unmanaged_id() const
    {
        return naming::id_type(derived().get_base_gid(),
            naming::id_type::unmanaged);
    }

    void set_locality_id(std::uint32_t locality_id, error_code& ec = throws)
    {
        if (gid_) {
            HPX_THROWS_IF(ec, invalid_status,
                "fixed_component_base::set_locality_id",
                "can't change locality_id after GID has already been registered");
        }
        else {
            // erase current locality_id and replace with given one
            msb_ = naming::replace_locality_id(msb_, locality_id);
        }
    }

#if defined(HPX_DISABLE_ASSERTS) || defined(BOOST_DISABLE_ASSERTS) || defined(NDEBUG)
    static constexpr void mark_as_migrated()
    {
    }
    static constexpr void on_migrated()
    {
    }
#else
    void mark_as_migrated()
    {
        // If this assertion is triggered then this component instance is being
        // migrated even if the component type has not been enabled to support
        // migration.
        HPX_ASSERT(false);
    }

    void on_migrated()
    {
        // If this assertion is triggered then this component instance is being
        // migrated even if the component type has not been enabled to support
        // migration.
        HPX_ASSERT(false);
    }
#endif

private:
    mutable naming::gid_type gid_;
    std::uint64_t msb_;
    std::uint64_t lsb_;
};

namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    struct fixed_heap
    {
#if defined(HPX_DISABLE_ASSERTS) || defined(BOOST_DISABLE_ASSERTS) || defined(NDEBUG)
        static constexpr void* alloc(std::size_t /*count*/)
        {
            return nullptr;
        }
        static constexpr void free(void* /*p*/, std::size_t /*count*/)
        {
        }
#else
        static void* alloc(std::size_t /*count*/)
        {
            HPX_ASSERT(false);        // this shouldn't ever be called
            return nullptr;
        }
        static void free(void* /*p*/, std::size_t /*count*/)
        {
            HPX_ASSERT(false);        // this shouldn't ever be called
        }
#endif
    };
}

///////////////////////////////////////////////////////////////////////////
template <typename Component>
class fixed_component : public Component
{
  public:
    typedef Component type_holder;
    typedef fixed_component<Component> component_type;
    typedef component_type derived_type;
    typedef detail::fixed_heap heap_type;

#if defined(HPX_DISABLE_ASSERTS) || defined(BOOST_DISABLE_ASSERTS) || defined(NDEBUG)
    static constexpr Component* create(std::size_t /* count */)
    {
        return nullptr;
    }

    static constexpr void destroy(
        Component* /* p */, std::size_t /* count */ = 1)
    {
    }
#else
    /// \brief  The function \a create is used for allocation and
    ///         initialization of instances of the derived components.
    static Component* create(std::size_t /* count */)
    {
        HPX_ASSERT(false);        // this shouldn't ever be called
        return nullptr;
    }

    /// \brief  The function \a destroy is used for destruction and
    ///         de-allocation of instances of the derived components.
    static void destroy(Component* /* p */, std::size_t /* count */ = 1)
    {
        HPX_ASSERT(false);        // this shouldn't ever be called
    }
#endif
};

}}


