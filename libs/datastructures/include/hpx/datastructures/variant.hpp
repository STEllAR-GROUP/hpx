//  Copyright (c) 2017-2019 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_DATASTRUCTURES_VARIANT_HPP)
#define HPX_DATASTRUCTURES_VARIANT_HPP

#include <hpx/config.hpp>
#include <hpx/datastructures/detail/variant.hpp>

namespace hpx { namespace util
{
    using mpark::variant;
    using mpark::monostate;

    using mpark::holds_alternative;
    using mpark::get;
    using mpark::get_if;
    using mpark::visit;
}}

#endif
