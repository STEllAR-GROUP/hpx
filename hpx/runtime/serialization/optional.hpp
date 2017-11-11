//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_OPTIONAL_SERIALIZATION_HPP)
#define HPX_OPTIONAL_SERIALIZATION_HPP

#include <hpx/config.hpp>
#include <hpx/util/optional.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

namespace hpx { namespace serialization
{
    template <typename T>
    void save(output_archive& ar, hpx::util::optional<T> const& o, unsigned)
    {
        bool const valid = bool(o);
        ar << valid;
        if (valid)
        {
            ar << *o;
        }
    }

    template <typename T>
    void load(input_archive& ar, hpx::util::optional<T>& o, unsigned)
    {
        bool valid = false;
        ar >> valid;
        if (!valid)
        {
            o.reset();
            return;
        }

        T value;
        ar >> value;
        o.emplace(std::move(value));
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(
        (template <typename T>), (hpx::util::optional<T>));
}}

#endif // HPX_OPTIONAL_SERIALIZATION_HPP
