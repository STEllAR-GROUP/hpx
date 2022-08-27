//  copyright (c) 2005
//  troy d. straszheim <troy@resophonic.com>
//  http://www.resophonic.com
//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_SERIALIZATION_HAVE_BOOST_TYPES)

#include <hpx/modules/errors.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <utility>

#include <boost/variant.hpp>

namespace hpx { namespace serialization {

    namespace detail {

        ////////////////////////////////////////////////////////////////////////
        struct boost_variant_save_visitor : boost::static_visitor<>
        {
            boost_variant_save_visitor(output_archive& ar)
              : m_ar(ar)
            {
            }

            template <typename T>
            void operator()(T const& value) const
            {
                m_ar << value;
            }

        private:
            output_archive& m_ar;
        };

        ////////////////////////////////////////////////////////////////////////
        template <typename... Ts>
        struct boost_variant_impl;

        template <typename T, typename... Ts>
        struct boost_variant_impl<T, Ts...>
        {
            template <typename V>
            static void load(input_archive& ar, int which, V& v)
            {
                if (which == 0)
                {
                    T value;
                    ar >> value;
                    v = HPX_MOVE(value);
                    return;
                }
                boost_variant_impl<Ts...>::load(ar, which - 1, v);
            }
        };

        template <>
        struct boost_variant_impl<>
        {
            template <typename V>
            static void load(input_archive& /*ar*/, int /*which*/, V& /*v*/)
            {
            }
        };
    }    // namespace detail

    template <typename... T>
    void save(output_archive& ar, boost::variant<T...> const& v, unsigned)
    {
        int which = v.which();
        ar << which;
        detail::boost_variant_save_visitor visitor(ar);
        v.apply_visitor(visitor);
    }

    template <typename... T>
    void load(input_archive& ar, boost::variant<T...>& v, unsigned)
    {
        int which;
        ar >> which;
        if (which >= static_cast<int>(sizeof...(T)))
        {
            // this might happen if a type was removed from the list of variant
            // types
            HPX_THROW_EXCEPTION(serialization_error,
                "load<Archive, Variant, version>",
                "type was removed from the list of variant types");
        }
        detail::boost_variant_impl<T...>::load(ar, which, v);
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(
        (template <typename... T>), (boost::variant<T...>) )
}}    // namespace hpx::serialization

#endif    // HPX_SERIALIZATION_HAVE_BOOST_TYPES
