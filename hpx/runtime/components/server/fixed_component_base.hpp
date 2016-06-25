//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_F5D19D10_9D74_4DB9_9ABB_ECCF2FA54497)
#define HPX_F5D19D10_9D74_4DB9_9ABB_ECCF2FA54497

#include <hpx/config.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/bind_naming_wrappers.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/create_component_fwd.hpp>
#include <hpx/runtime/components_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/is_component.hpp>
#include <hpx/util/unique_function.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>

#include <sstream>

namespace hpx { namespace components
{

template <typename Component>
class fixed_component;

///////////////////////////////////////////////////////////////////////////
template <typename Component>
class fixed_component_base : public traits::detail::fixed_component_tag
{
private:
    typedef typename boost::mpl::if_<
            boost::is_same<Component, detail::this_type>,
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
    fixed_component_base(boost::uint64_t msb, boost::uint64_t lsb)
      : msb_(msb)
      , lsb_(lsb)
    {}

    ~fixed_component_base()
    {}

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

    // \brief This exposes the component type.
    static component_type get_component_type()
    {
        return components::get_component_type<this_component_type>();
    }
    static void set_component_type(component_type type)
    {
        components::set_component_type<this_component_type>(type);
    }

private:
    // declare friends which are allowed to access get_base_gid()
    template <typename Component_>
    friend naming::gid_type server::create(std::size_t count);

    template <typename Component_>
    friend naming::gid_type server::create(
        util::unique_function_nonser<void(void*)> const& ctor);

    template <typename Component_>
    friend naming::gid_type server::create(naming::gid_type const& gid,
        util::unique_function_nonser<void(void*)> const& ctor);

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

#if defined(HPX_HAVE_COMPONENT_GET_GID_COMPATIBILITY)
    naming::id_type get_gid() const
    {
        return get_unmanaged_id();
    }
#endif

    void set_locality_id(boost::uint32_t locality_id, error_code& ec = throws)
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

#if defined(HPX_HAVE_SECURITY)
    static components::security::capability get_required_capabilities(
        components::security::traits::capability<>::capabilities caps)
    {
        return components::default_component_creation_capabilities(caps);
    }
#endif

    // Pinning functionality
    void pin() {}
    void unpin() {}
    boost::uint32_t pin_count() const { return 0; }

    void mark_as_migrated()
    {
        // If this assertion is triggered then this component instance is being
        // migrated even if the component type has not been enabled to support
        // migration.
        HPX_ASSERT(false);
    }

private:
    mutable naming::gid_type gid_;
    boost::uint64_t msb_;
    boost::uint64_t lsb_;
};

namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    template <typename Component>
    struct fixed_heap_factory
    {
        static Component* alloc(std::size_t count)
        {
            HPX_ASSERT(false);        // this shouldn't ever be called
            return 0;
        }
        static void free(void* p, std::size_t count)
        {
            HPX_ASSERT(false);        // this shouldn't ever be called
        }
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
    typedef detail::fixed_heap_factory<component_type> heap_type;

    /// \brief  The function \a create is used for allocation and
    ///         initialization of instances of the derived components.
    static Component* create(std::size_t count)
    {
        HPX_ASSERT(false);        // this shouldn't ever be called
        return 0;
    }

    /// \brief  The function \a destroy is used for destruction and
    ///         de-allocation of instances of the derived components.
    static void destroy(Component* p, std::size_t count = 1)
    {
        HPX_ASSERT(false);        // this shouldn't ever be called
    }
};

}}

#endif // HPX_F5D19D10_9D74_4DB9_9ABB_ECCF2FA54497

