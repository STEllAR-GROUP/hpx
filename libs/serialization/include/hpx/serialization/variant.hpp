//  copyright (c) 2005
//  troy d. straszheim <troy@resophonic.com>
//  http://www.resophonic.com
//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/boost_variant.hpp>    // for backwards compatibility

#if defined(HPX_HAVE_CXX17_STD_VARIANT)

#include <hpx/datastructures/variant_helper.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <cstddef>
#include <utility>
#include <variant>

namespace hpx { namespace serialization {

    namespace detail {

        ////////////////////////////////////////////////////////////////////////
        struct std_variant_save_visitor
        {
            std_variant_save_visitor(output_archive& ar)
              : ar_(ar)
            {
            }

            template <typename T>
            void operator()(T const& value) const
            {
                ar_ << value;
            }

        private:
            output_archive& ar_;
        };

        ////////////////////////////////////////////////////////////////////////
        template <typename... Ts>
        struct std_variant_impl;

        template <typename T, typename... Ts>
        struct std_variant_impl<T, Ts...>
        {
            template <typename V>
            static void load(input_archive& ar, std::size_t which, V& v)
            {
                if (which == 0)
                {
                    T value;
                    ar >> value;
                    v.template emplace<T>(std::move(value));
                    return;
                }
                std_variant_impl<Ts...>::load(ar, which - 1, v);
            }
        };

        template <>
        struct std_variant_impl<>
        {
            template <typename V>
            static void load(
                input_archive& /*ar*/, std::size_t /*which*/, V& /*v*/)
            {
            }
        };
    }    // namespace detail

    template <typename... Ts>
    void save(output_archive& ar, std::variant<Ts...> const& v, unsigned)
    {
        std::size_t which = v.index();
        ar << which;
        detail::std_variant_save_visitor visitor(ar);
        std::visit(visitor, v);
    }

    template <typename... Ts>
    void load(input_archive& ar, std::variant<Ts...>& v, unsigned)
    {
        std::size_t which;
        ar >> which;
        if (which >= sizeof...(Ts))
        {
            // this might happen if a type was removed from the list of variant
            // types
            HPX_THROW_EXCEPTION(serialization_error,
                "load<Archive, Variant, version>",
                "type was removed from the list of variant types");
        }
        detail::std_variant_impl<Ts...>::load(ar, which, v);
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(
        (template <typename... Ts>), (std::variant<Ts...>) );
}}    // namespace hpx::serialization

#endif    // HPX_HAVE_CXX17_STD_VARIANT
