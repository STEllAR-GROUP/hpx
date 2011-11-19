//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_HANDLE_GID_OCT_27_2011_0418PM)
#define HPX_TRAITS_HANDLE_GID_OCT_27_2011_0418PM

#include <hpx/traits.hpp>

namespace hpx { namespace traits
{
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
    template <typename T, typename F, typename Enable>
    struct handle_gid
    {
        static bool call(T const&, F)
        {
            return true;    // do nothing for arbitrary types
        }
    };

    template <typename F>
    struct handle_gid<naming::id_type, F>
    {
        static bool call(naming::id_type const &id, F const& f)
        {
            f(boost::ref(id));
            return true;
        }
    };

    template <typename F>
    struct handle_gid<std::vector<naming::id_type>, F>
    {
        static bool call(std::vector<naming::id_type> const& ids, F const& f)
        {
            BOOST_FOREACH(naming::id_type const& id, ids)
                f(boost::ref(id));
            return true;
        }
    };
}}

#endif
