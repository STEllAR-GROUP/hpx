//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_GET_RESULT_FEB_10_2011_1123AM)
#define HPX_LCOS_GET_RESULT_FEB_10_2011_1123AM

namespace hpx { namespace lcos
{
    template <typename Result, typename RemoteResult>
    struct get_result
    {
        static Result call(RemoteResult const& rhs)
        {
            return Result(rhs);
        }
    };

    template <typename Result>
    struct get_result<Result, Result>
    {
        static Result const& call(Result const& rhs)
        {
            return rhs;
        }
    };
}}

#endif
