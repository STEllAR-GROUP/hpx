//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_ACTIONS_DETAIL_REMOTE_ACTION_RESULT_HPP
#define HPX_RUNTIME_ACTIONS_DETAIL_REMOTE_ACTION_RESULT_HPP

namespace hpx { namespace actions { namespace detail
{
    template <typename Result>
    struct remote_action_result
    {
        typedef Result type;
    };

}}}

#endif
