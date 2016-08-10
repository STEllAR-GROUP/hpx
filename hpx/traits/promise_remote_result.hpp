//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_PROMISE_REMOTE_RESULT_OCT_27_2011_0416PM)
#define HPX_TRAITS_PROMISE_REMOTE_RESULT_OCT_27_2011_0416PM

#include <hpx/util/unused.hpp>

namespace hpx { namespace traits
{
    template <typename Result, typename Enable = void>
    struct promise_remote_result
    {
        typedef Result type;
    };

    template <>
    struct promise_remote_result<void>
    {
        typedef hpx::util::unused_type type;
    };
}}

#endif
