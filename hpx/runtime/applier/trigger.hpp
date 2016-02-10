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
    naming::id_type const& k
  , Arg0 && arg0
    )
{
    set_lco_value(k, std::forward<Arg0>(arg0));
}

inline void trigger(
    naming::id_type const& k
    )
{
    trigger_lco_event(k);
}

inline void trigger_error(
    naming::id_type const& k
  , boost::exception_ptr const& e
    )
{
    set_lco_error(k, e);
}

inline void trigger_error(
    naming::id_type const& k
  , boost::exception_ptr && e
    )
{
    set_lco_error(k, e);
}

}}

#endif // HPX_6B2240CE_5CE8_43EA_BAFF_5C8F17D21AAE

