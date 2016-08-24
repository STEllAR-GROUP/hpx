//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_SERIALIZABLE_FUNCTION_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_SERIALIZABLE_FUNCTION_VTABLE_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/access.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/util/detail/empty_function.hpp>
#include <hpx/util/detail/function_registration.hpp>
#include <hpx/util/detail/vtable/serializable_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

#include <type_traits>

namespace hpx { namespace util { namespace detail
{
    template <typename Function>
    struct init_registration;

    template <typename VTable, typename T>
    struct serializable_function_registration;

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable>
    struct serializable_function_vtable
      : VTable, serializable_vtable
    {
        char const* name;

        template <typename T>
        HPX_CONSTEXPR serializable_function_vtable(construct_vtable<T>) HPX_NOEXCEPT
          : VTable(construct_vtable<T>())
          , serializable_vtable(construct_vtable<T>())
          , name((init_registration<
                    serializable_function_registration<VTable, T>
                >::g.register_function(), this->empty
                  ? "empty" : get_function_name<
                        serializable_function_registration<VTable, T>
                    >()))
        {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename T>
    struct serializable_function_registration
    {
        typedef serializable_function_vtable<VTable> first_type;
        typedef T second_type;
    };

    ///////////////////////////////////////////////////////////////////////////
    // registration code for serialization
    template <typename VTable, typename T>
    struct init_registration<
        serializable_function_registration<VTable, T>
    >
    {
        static automatic_function_registration<
            serializable_function_registration<VTable, T>
        > g;
    };

    template <typename VTable, typename T>
    automatic_function_registration<
        serializable_function_registration<VTable, T>
    > init_registration<
        serializable_function_registration<VTable, T>
    >::g =  automatic_function_registration<
                serializable_function_registration<VTable, T>
            >();
}}}

// Pseudo registration for empty functions.
// We don't want to serialize empty functions.
namespace hpx { namespace util { namespace detail
{
    template <typename VTable, typename Sig>
    struct get_function_name_impl<
        hpx::util::detail::serializable_function_registration<
            VTable
          , hpx::util::detail::empty_function<Sig>
        >
    >
    {
        static char const * call()
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "get_function_name<empty_function>");
            return "";
        }
    };
}}}

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename Sig>
    struct needs_automatic_registration<
        hpx::util::detail::serializable_function_registration<
            VTable
          , hpx::util::detail::empty_function<Sig>
        >
    > : std::false_type
    {};
}}

#endif
