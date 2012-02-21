//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_OCT_26_2011_0838AM)
#define HPX_TRAITS_OCT_26_2011_0838AM

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename Enable = void>
    struct promise_remote_result;

    template <typename Result, typename Enable = void>
    struct promise_local_result;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult, typename Enable = void>
    struct get_remote_result;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Enable = void>
    struct component_type_database;
}}

#endif
