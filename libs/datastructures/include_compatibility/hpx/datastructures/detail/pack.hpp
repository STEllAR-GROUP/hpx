//  Copyright (c) 2019 Auriane Reverdell
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/datastructures/config/defines.hpp>
#include <hpx/type_support/pack.hpp>

#if defined(HPX_DATASTRUCTURES_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/datastructures/detail/pack.hpp is deprecated, \
    please include hpx/type_support/pack.hpp instead")
#else
#warning "The header hpx/datastructures/detail/pack.hpp is deprecated, \
    please include hpx/type_support/pack.hpp instead"
#endif
#endif

namespace hpx { namespace util { namespace detail {

    using hpx::util::pack;
    using hpx::util::pack_c;

    ///////////////////////////////////////////////////////////////////////////
    using hpx::util::make_index_pack;

    ///////////////////////////////////////////////////////////////////////////
    using hpx::util::all_of;
    using hpx::util::any_of;
    using hpx::util::contains;
    using hpx::util::none_of;

    ///////////////////////////////////////////////////////////////////////////
    using hpx::util::at_index;
}}}    // namespace hpx::util::detail
