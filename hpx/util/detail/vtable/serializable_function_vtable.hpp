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
#include <hpx/util/detail/empty_function.hpp>
#include <hpx/util/detail/vtable/serializable_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

#include <string>
#include <type_traits>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable>
    struct serializable_function_vtable
      : serializable_vtable
    {
        VTable const* vptr;
        char const* name;

        template <typename T>
        serializable_function_vtable(construct_vtable<T>) noexcept
          : serializable_vtable(construct_vtable<T>())
          , vptr(detail::get_vtable<VTable, T>())
          , name(detail::get_function_name<VTable, T>())
        {
            hpx::serialization::detail::polymorphic_intrusive_factory::instance().
                register_class(name, &serializable_function_vtable::get_vtable<T>);
        }

        serializable_function_vtable(construct_vtable<empty_function>) noexcept
          : serializable_vtable(construct_vtable<empty_function>())
          , vptr(detail::get_empty_function_vtable<VTable>())
          , name("empty")
        {}

        template <typename T>
        static void* get_vtable()
        {
            typedef serializable_function_vtable<VTable> vtable_type;
            return const_cast<vtable_type*>(detail::get_vtable<vtable_type, T>());
        }
    };

    template <typename VTable>
    serializable_function_vtable<VTable> const*
    get_serializable_vtable(std::string const& name)
    {
        using serializable_vtable = serializable_function_vtable<VTable>;
        return
            hpx::serialization::detail::polymorphic_intrusive_factory::instance().
                create<serializable_vtable const>(name);
    }
}}}

#endif
