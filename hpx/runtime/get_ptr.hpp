//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_GET_PTR_SEP_18_2013_0622PM)
#define HPX_RUNTIME_GET_PTR_SEP_18_2013_0622PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>

#include <boost/shared_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        struct get_ptr_deleter
        {
            get_ptr_deleter(naming::id_type id) : id_(id) {}

            template <typename Component>
            void operator()(Component*)
            {
                id_ = naming::invalid_id;       // release component
            }

            naming::id_type id_;                // hold component alive
        };
    }

    template <typename Component>
    boost::shared_ptr<Component> get_ptr(naming::id_type id, error_code& ec = throws)
    {
        naming::resolver_client& agas = naming::get_agas_client();
        naming::address addr;
        if (!agas.resolve(id, addr, ec) || ec)
        {
            HPX_THROWS_IF(ec, bad_parameter, "hpx::get_ptr",
                "can't resolve the given component id");
            return boost::shared_ptr<Component>();
        }

        if (get_locality() != addr.locality_)
        {
            HPX_THROWS_IF(ec, bad_parameter, "hpx::get_ptr",
                "the given component id does not belong to a local object");
            return boost::shared_ptr<Component>();
        }

        if (!components::types_are_compatible(addr.type_,
                components::get_component_type<Component>()))
        {
            HPX_THROWS_IF(ec, bad_component_type, "hpx::get_ptr",
                "requested component type does not match the given component id");
            return boost::shared_ptr<Component>();
        }

        Component* p = get_lva<Component>::call(addr.address_);
        return boost::shared_ptr<Component>(p, detail::get_ptr_deleter(id));
    }
}

#endif
