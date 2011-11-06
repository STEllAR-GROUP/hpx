////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_6B2240CE_5CE8_43EA_BAFF_5C8F17D21AAE)
#define HPX_6B2240CE_5CE8_43EA_BAFF_5C8F17D21AAE

#include <hpx/runtime/actions/continuation.hpp>

namespace hpx { namespace applier
{

template <typename Arg0>
inline void trigger(
    naming::id_type k
  , Arg0 const& arg0
    )
{
    return actions::continuation(k).trigger<Arg0>(arg0);
}

template <typename Arg0>
inline void trigger(
    naming::id_type k
  , BOOST_FWD_REF(Arg0) arg0
    )
{
    return actions::continuation(k).trigger<Arg0>(boost::forward<Arg0>(arg0));
}

inline void trigger(
    naming::id_type const& k
    )
{
    actions::continuation(k).trigger();
}

inline void trigger_error(
    naming::id_type const& k
  , boost::exception_ptr const& e
    )
{
    actions::continuation(k).trigger_error(e);
}

inline void trigger_error(
    naming::id_type const& k
  , BOOST_RV_REF(boost::exception_ptr) e
    )
{
    actions::continuation(k).trigger_error(boost::move(e));
}

}}

#endif // HPX_6B2240CE_5CE8_43EA_BAFF_5C8F17D21AAE

