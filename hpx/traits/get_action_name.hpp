//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_GET_ACTION_NAME_OCT_27_2011_0420PM)
#define HPX_TRAITS_GET_ACTION_NAME_OCT_27_2011_0420PM

#include <hpx/traits.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Enable>
    struct get_action_name
    {
        static HPX_ALWAYS_EXPORT char const* call();
    };
}}

#endif
