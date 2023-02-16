//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/detail/polymorphic_nonintrusive_factory.hpp>

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/string.hpp>

#include <string>

namespace hpx::serialization::detail {

    template <typename T>
    void polymorphic_nonintrusive_factory::save(output_archive& ar, T const& t)
    {
        // It's safe to call typeid here. The typeid(t) return value is
        // only used for local lookup to the portable string that goes over the
        // wire
        std::string const class_name = typeinfo_map_.at(typeid(t).name());
        ar << class_name;

        map_.at(class_name).save_function(ar, &t);
    }

    template <typename T>
    void polymorphic_nonintrusive_factory::load(input_archive& ar, T& t)
    {
        std::string class_name;
        ar >> class_name;

        map_.at(class_name).load_function(ar, &t);
    }

    template <typename T>
    T* polymorphic_nonintrusive_factory::load(input_archive& ar)
    {
        std::string class_name;
        ar >> class_name;

        function_bunch_type const& bunch = map_.at(class_name);
        T* t = static_cast<T*>(bunch.create_function(ar));

        return t;
    }
}    // namespace hpx::serialization::detail
