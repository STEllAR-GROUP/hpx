//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_EXTRACT_PARTITIONER_OCT_03_2014_0105PM)
#define HPX_PARALLEL_TRAITS_EXTRACT_PARTITIONER_OCT_03_2014_0105PM

#include <hpx/config.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    struct static_partitioner_tag {};
    struct auto_partitioner_tag {};
    struct default_partitioner_tag {};

    template <typename ExPolicy, typename Enable = void>
    struct extract_partitioner
    {
        typedef default_partitioner_tag type;
    };
}}}

#endif
