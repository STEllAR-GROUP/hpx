////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_CDD12289_0A65_47A4_BC53_A4670CDAF5A7)
#define HPX_CDD12289_0A65_47A4_BC53_A4670CDAF5A7

#include <vector>

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

namespace hpx { namespace test { namespace server
{

struct HPX_COMPONENT_EXPORT simple_refcnt_checker
  : components::simple_component_base<simple_refcnt_checker>
{
  private:
    naming::id_type target_;
    std::vector<naming::id_type> references_;

  public:
    simple_refcnt_checker()
      : target_(naming::invalid_id)
      , references_()
    {}

    simple_refcnt_checker(
        naming::id_type const& target
        )
      : target_(target)
      , references_()
    {}

    ~simple_refcnt_checker();

    void take_reference(
        naming::id_type const& gid
        )
    {
        references_.push_back(gid);
    }

    HPX_DEFINE_COMPONENT_ACTION(simple_refcnt_checker, take_reference);
};

}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::test::server::simple_refcnt_checker::take_reference_action,
    simple_refcnt_checker_take_reference_action);

#endif // HPX_CDD12289_0A65_47A4_BC53_A4670CDAF5A7

