//  Copyright (c) 2015-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nominmax

#if !defined(HPX_PARALLEL_CONTAINER_ALGORITHM_TAGSPEC_DEC_23_2015_1156AM)
#define HPX_PARALLEL_CONTAINER_ALGORITHM_TAGSPEC_DEC_23_2015_1156AM

#include <hpx/config.hpp>
#include <hpx/util/tagged.hpp>

namespace hpx { namespace parallel { inline namespace v1
{
    HPX_DEFINE_TAG_SPECIFIER(in)        // defines tag::in
    HPX_DEFINE_TAG_SPECIFIER(out)       // defines tag::out
    HPX_DEFINE_TAG_SPECIFIER(begin)     // defines tag::begin
    HPX_DEFINE_TAG_SPECIFIER(end)       // defines tag::end
    HPX_DEFINE_TAG_SPECIFIER(in1)       // defines tag::in1
    HPX_DEFINE_TAG_SPECIFIER(in2)       // defines tag::in2

#if defined(HPX_MSVC)
#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max
#endif

    HPX_DEFINE_TAG_SPECIFIER(min)       // defines tag::min
    HPX_DEFINE_TAG_SPECIFIER(max)       // defines tag::max

#if defined(HPX_MSVC)
#pragma pop_macro("min")
#pragma pop_macro("max")
#endif
}}}

#endif


