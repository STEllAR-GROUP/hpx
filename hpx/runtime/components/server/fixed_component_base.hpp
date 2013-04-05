//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_F5D19D10_9D74_4DB9_9ABB_ECCF2FA54497)
#define HPX_F5D19D10_9D74_4DB9_9ABB_ECCF2FA54497

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/bind_naming_wrappers.hpp>
#include <hpx/util/stringstream.hpp>

namespace hpx { namespace detail
{
    HPX_API_EXPORT naming::gid_type get_next_id();
}}

namespace hpx { namespace components
{

template <typename Component>
class fixed_component;

///////////////////////////////////////////////////////////////////////////
template <typename Component>
class fixed_component_base : public detail::fixed_component_tag
{
private:
    typedef typename boost::mpl::if_<
            boost::is_same<Component, detail::this_type>,
            fixed_component_base, Component
        >::type this_component_type;

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
            applier::unbind_gid(gid_, ec);
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

    /// \brief Return the component's fixed GID.
    ///
    /// \returns The fixed global id (GID) for this component
    naming::gid_type get_base_gid() const
    {
        if (!gid_)
        {
            naming::address addr(applier::get_applier().here(),
                components::get_component_type<wrapped_type>(),
                std::size_t(static_cast<this_component_type const*>(this)));

            gid_ = naming::gid_type(msb_, lsb_);

            // Try to bind the preset GID first
            if (!applier::bind_gid(gid_, addr))
            {
                gid_ = hpx::detail::get_next_id();

                // If we can't bind the preset GID, then try to bind the next
                // available GID on this locality.
                if (!applier::bind_gid(gid_, addr))
                {
                    hpx::util::osstream strm;
                    strm << gid_;
                    gid_ = naming::gid_type();   // invalidate GID
                    HPX_THROW_EXCEPTION(duplicate_component_address,
                        "fixed_component_base<Component>::get_base_gid",
                        hpx::util::osstream_get_string(strm));
                }
            }
        }
        return gid_;
    }

    naming::id_type get_gid() const
    {
        return naming::id_type(get_base_gid(), naming::id_type::unmanaged);
    }

    /// \brief  The function \a get_factory_properties is used to
    ///         determine, whether instances of the derived component can
    ///         be created in blocks (i.e. more than one instance at once).
    ///         This function is used by the \a distributing_factory to
    ///         determine a correct allocation strategy
    static factory_property get_factory_properties()
    {
        // components derived from this template have to be allocated one
        // at a time
        return factory_none;
    }

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
            BOOST_ASSERT(false);        // this shouldn't ever be called
            return 0;
        }
        static void free(void* p, std::size_t count)
        {
            BOOST_ASSERT(false);        // this shouldn't ever be called
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
        BOOST_ASSERT(false);        // this shouldn't ever be called
        return 0;
    }

    /// \brief  The function \a destroy is used for destruction and
    ///         de-allocation of instances of the derived components.
    static void destroy(Component* p, std::size_t count = 1)
    {
        BOOST_ASSERT(false);        // this shouldn't ever be called
    }
};

}}

#endif // HPX_F5D19D10_9D74_4DB9_9ABB_ECCF2FA54497

