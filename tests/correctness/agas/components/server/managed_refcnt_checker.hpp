////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_CDD12289_0A65_47A4_BC53_A4670CDAF5A7)
#define HPX_CDD12289_0A65_47A4_BC53_A4670CDAF5A7

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/constructor_argument.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>

namespace hpx { namespace test { namespace server
{

struct HPX_COMPONENT_EXPORT managed_refcnt_checker
  : components::managed_component_base<managed_refcnt_checker>
{
  private:
    naming::id_type target_;

  public:
    managed_refcnt_checker()
      : target_(naming::invalid_id)
    {}

    managed_refcnt_checker(
        components::constructor_argument const& target_
        )
      : target_(boost::get<naming::id_type>(target_))
    {}

    ~managed_refcnt_checker();
};

}}}

#endif // HPX_CDD12289_0A65_47A4_BC53_A4670CDAF5A7

