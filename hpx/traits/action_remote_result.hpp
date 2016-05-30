//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_ACTION_REMOTE_RESULT_HPP
#define HPX_TRAITS_ACTION_REMOTE_RESULT_HPP

namespace hpx { namespace traits
{
    namespace detail
    {
        template <typename Result, typename Enable = void>
        struct action_remote_result_customization_point
        {
            typedef Result type;
        };
    }

    template <typename Result>
    struct action_remote_result
      : detail::action_remote_result_customization_point<Result>
    {};
}}

#endif
