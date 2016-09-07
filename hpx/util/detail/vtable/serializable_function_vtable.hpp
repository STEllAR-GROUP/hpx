//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_SERIALIZABLE_FUNCTION_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_SERIALIZABLE_FUNCTION_VTABLE_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_intrusive_factory.hpp>
#include <hpx/util/detail/function_registration.hpp>
#include <hpx/util/detail/vtable/serializable_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

#include <string>
#include <type_traits>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable>
    struct serializable_function_vtable
      : VTable, serializable_vtable
    {
        char const* name;

        template <typename T>
        serializable_function_vtable(construct_vtable<T>) HPX_NOEXCEPT
          : VTable(construct_vtable<T>())
          , serializable_vtable(construct_vtable<T>())
          , name(this->empty ? "empty" : get_function_name<VTable, T>())
        {
            hpx::serialization::detail::polymorphic_intrusive_factory::instance().
                register_class(name, &serializable_function_vtable::get_vtable<T>);
        }

        template <typename T>
        static void* get_vtable()
        {
            typedef serializable_function_vtable<VTable> vtable_type;
            return const_cast<vtable_type*>(detail::get_vtable<vtable_type, T>());
        }
    };

    template <typename VTable>
    VTable const* get_vtable(std::string const& name)
    {
        return
            hpx::serialization::detail::polymorphic_intrusive_factory::instance().
                create<VTable const>(name);
    }
}}}

#endif
