//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013-2023 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/functional/detail/empty_function.hpp>
#include <hpx/functional/detail/function_registration.hpp>
#include <hpx/functional/detail/vtable/vtable.hpp>
#include <hpx/functional/serialization/detail/vtable/serializable_vtable.hpp>
#include <hpx/serialization/detail/polymorphic_intrusive_factory.hpp>

#include <string>
#include <type_traits>

namespace hpx::util::detail {

    template <typename VTable>
    struct serializable_function_vtable;

    template <typename VTable, typename T>
    struct serializable_vtables
    {
        static serializable_function_vtable<VTable> const instance;
    };

    template <typename VTable, typename T>
    serializable_function_vtable<VTable> const
        serializable_vtables<VTable, T>::instance =
            serializable_function_vtable<VTable>(detail::construct_vtable<T>());

    template <typename VTable, typename T>
    constexpr serializable_function_vtable<VTable> const*
    get_serializable_vtable() noexcept
    {
        static_assert(std::is_same_v<T, std::decay_t<T>>,
            "T shall have no cv-ref-qualifiers");

        return &serializable_vtables<VTable, T>::instance;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable>
    struct serializable_function_vtable : serializable_vtable
    {
        VTable const* vptr;
        char const* name;

        template <typename T>
        explicit serializable_function_vtable(construct_vtable<T>) noexcept
          : serializable_vtable(construct_vtable<T>())
          , vptr(detail::get_vtable<VTable, T>())
          , name(detail::get_function_name<VTable, T>())
        {
            hpx::serialization::detail::polymorphic_intrusive_factory::
                instance()
                    .register_class(
                        name, &serializable_function_vtable::get<T>);
        }

        explicit serializable_function_vtable(
            construct_vtable<empty_function>) noexcept
          : serializable_vtable(construct_vtable<empty_function>())
          , vptr(detail::get_empty_function_vtable<VTable>())
          , name("empty")
        {
        }

        template <typename T>
        static void* get()
        {
            using vtable_type = serializable_function_vtable<VTable>;
            return const_cast<vtable_type*>(
                detail::get_serializable_vtable<VTable, T>());
        }
    };

    template <typename VTable>
    [[nodiscard]] serializable_function_vtable<VTable> const*
    get_serializable_vtable(std::string const& name)
    {
        using serializable_vtable = serializable_function_vtable<VTable>;
        return hpx::serialization::detail::polymorphic_intrusive_factory::
            instance()
                .create<serializable_vtable const>(name);
    }
}    // namespace hpx::util::detail
