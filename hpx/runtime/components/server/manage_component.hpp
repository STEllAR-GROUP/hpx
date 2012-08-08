//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_COMPONENTS_SERVER_MANAGE_COMPONENT_JUN_02_2008_0146PM)
#define HPX_COMPONENTS_SERVER_MANAGE_COMPONENT_JUN_02_2008_0146PM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/util/stringstream.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_COMPONENT_CREATE_ARGUMENT_LIMIT,                              \
    "hpx/runtime/components/server/manage_component.hpp"))                    \
    /**/

#include BOOST_PP_ITERATE()

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// Create arrays of components using their default constructor
    template <typename Component>
    naming::gid_type create (std::size_t count, error_code& ec = throws)
    {
        if (0 == count) {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "create<Component>", "count shouldn't be zero");
            return naming::invalid_gid;
        }

        Component* c = static_cast<Component*>(Component::create(count));
        naming::gid_type gid = c->get_base_gid();
        if (gid) {
            // register the new object in the local AGAS cache
            naming::resolver_client& cl = naming::get_agas_client();
            cl.update_cache(gid, cl.get_here(),
                components::get_component_type<typename Component::wrapped_type>(),
                reinterpret_cast<boost::uint64_t>(c), count, ec);

            // everything is ok, return the new id
            if (&ec != &throws)
                ec = make_success_code();
            return gid;
        }

        delete c;

        hpx::util::osstream strm;
        strm << "global id " << gid << " is already bound to a different "
                "component instance";
        HPX_THROWS_IF(ec, hpx::duplicate_component_address,
            "create<Component>",
            hpx::util::osstream_get_string(strm));

        return naming::invalid_gid;
    }

    template <typename Component>
    naming::gid_type create_one (HPX_STD_FUNCTION<void(void*)> const& ctor,
        error_code& ec = throws)
    {
        Component* c = Component::heap_type::alloc(1);

        ctor(c);
        naming::gid_type gid = c->get_base_gid();
        if (gid) {
            // register the new object in the local AGAS cache
            naming::resolver_client& cl = naming::get_agas_client();
            cl.update_cache(gid, cl.get_here(),
                components::get_component_type<typename Component::wrapped_type>(),
                reinterpret_cast<boost::uint64_t>(c), 1, ec);

            // everything is ok, return the new id
            if (&ec != &throws)
                ec = make_success_code();
            return gid;
        }

        Component::heap_type::free(c, 1);

        hpx::util::osstream strm;
        strm << "global id " << gid << " is already bound to a different "
                "component instance";
        HPX_THROWS_IF(ec, hpx::duplicate_component_address,
            "create_one<Component>(ctor)",
            hpx::util::osstream_get_string(strm));

        return naming::invalid_gid;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    void destroy(naming::gid_type const& gid, error_code& ec = throws)
    {
        // retrieve the local address bound to the given global id
        applier::applier& appl = hpx::applier::get_applier();
        naming::address addr;
        if (!appl.get_agas_client().resolve(gid, addr))
        {
            hpx::util::osstream strm;
            strm << "global id " << gid << " is not bound to any "
                    "component instance";
            HPX_THROWS_IF(ec, hpx::unknown_component_address,
                "destroy<Component>", hpx::util::osstream_get_string(strm));
            return;
        }

        // make sure this component is located here
        if (appl.here() != addr.locality_)
        {
            // FIXME: should the component be re-bound ?
            hpx::util::osstream strm;
            strm << "global id " << gid << " is not bound to any local "
                    "component instance";
            HPX_THROWS_IF(ec, hpx::unknown_component_address,
                "destroy<Component>", hpx::util::osstream_get_string(strm));
            return;
        }

        // make sure it's the correct component type
        components::component_type type =
            components::get_component_type<typename Component::wrapped_type>();
        if (!types_are_compatible(type, addr.type_))
        {
            // FIXME: should the component be re-bound ?
            hpx::util::osstream strm;
            strm << "global id " << gid << " is not bound to a component "
                    "instance of type: " << get_component_type_name(type)
                 << " (it is bound to a " << get_component_type_name(addr.type_)
                 << ")";
            HPX_THROWS_IF(ec, hpx::unknown_component_address,
                "destroy<Component>", hpx::util::osstream_get_string(strm));
            return;
        }

        // delete the local instances
        Component::destroy(reinterpret_cast<Component*>(addr.address_));
        if (&ec != &throws)
            ec = make_success_code();
    }
}}}

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// Create single instances of a component using additional constructor
    /// parameters
    // FIXME: error code?
    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename T)>
    naming::gid_type create_one(BOOST_PP_ENUM_BINARY_PARAMS(N, T, const& t))
    {
        Component* c = static_cast<Component*>(
            Component::create_one(BOOST_PP_ENUM_PARAMS(N, t)));
        naming::gid_type gid = c->get_base_gid();
        if (gid)
            return gid;

        delete c;

        hpx::util::osstream strm;
        strm << "global id " << gid << " is already bound to a different "
                "component instance";
        HPX_THROW_EXCEPTION(hpx::duplicate_component_address,
            "create_one<Component>",
            hpx::util::osstream_get_string(strm));

        return naming::invalid_gid;
    }

#undef N

}}}

#endif
