//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_OCT_26_2011_0838AM)
#define HPX_TRAITS_OCT_26_2011_0838AM

    /// \namespace traits
namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename Enable = void>
    struct promise_remote_result;

    template <typename Result, typename Enable = void>
    struct promise_local_result;

    ///////////////////////////////////////////////////////////////////////////
    // The customization point handle_gid is used to handle reference
    // counting of GIDs while they are transferred to a different locality.
    // It has to be specialized for arbitrary types, which may hold GIDs.
    //
    // It is important to make sure that all GID instances which are
    // contained in any transferred data structure are handled during
    // serialization. For this reason any user defined data type, which
    // is passed as an parameter to a action or which is returned from
    // a result_action needs to provide a corresponding specialization.
    //
    // The purpose of this customization point is to call the provided
    // function for all GIDs held in the data type.
    template <typename T, typename F, typename Enable = void>
    struct handle_gid;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Enable = void>
    struct get_action_name;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Enable = void>
    struct component_type_database;
}}

#endif
