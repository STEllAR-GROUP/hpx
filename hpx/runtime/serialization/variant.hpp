//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2017-2019 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_DATASTRUCTURES_VARIANT_SERIALIZATION_HPP)
#define HPX_DATASTRUCTURES_VARIANT_SERIALIZATION_HPP

#include <hpx/config.hpp>
#include <hpx/datastructures/variant.hpp>

#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/runtime/serialization/boost_variant.hpp>
#include <hpx/errors/throw_exception.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    void save(output_archive& ar, hpx::util::variant<Ts...> const& v, unsigned)
    {
        std::size_t which = v.index();
        ar << which;
        detail::variant_save_visitor visitor(ar);
        hpx::util::visit(visitor, v);
    }

    template <typename... Ts>
    void load(input_archive& ar, hpx::util::variant<Ts...>& v, unsigned)
    {
        std::size_t which;
        ar >> which;
        if (which >= sizeof...(Ts))
        {
            // this might happen if a type was removed from the list of variant
            // types
            HPX_THROW_EXCEPTION(serialization_error
              , "load<Archive, Variant, version>"
              , "type was removed from the list of variant types");
        }
        detail::variant_impl<Ts...>::load(ar, which, v);
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(
        (template <typename ... Ts>), (hpx::util::variant<Ts...>));
}}

#endif
