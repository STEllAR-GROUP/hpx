//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_NEEDS_GUID_INITIALIZATION_APR_15_2012_1207PM)
#define HPX_TRAITS_NEEDS_GUID_INITIALIZATION_APR_15_2012_1207PM

#include <hpx/traits.hpp>

namespace hpx { namespace traits
{
    // This trait is used to decide whether a class (or specialization) is
    // required to initialize its GUID for serialization.
    template <typename Action, typename Enable>
    struct needs_guid_initialization
      : boost::mpl::true_
    {};
}}

#endif

