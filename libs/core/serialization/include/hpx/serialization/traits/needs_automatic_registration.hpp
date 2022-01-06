//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#pragma once

#include <type_traits>

namespace hpx { namespace traits {

    // This trait is used to decide whether a class (or specialization) is
    // required to automatically register to the action factory
    template <typename T, typename Enable = void>
    struct needs_automatic_registration : std::true_type
    {
    };
}}    // namespace hpx::traits
