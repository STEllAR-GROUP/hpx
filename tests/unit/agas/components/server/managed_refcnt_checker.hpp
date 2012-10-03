////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_2F9C9286_A4A2_451C_BBD6_CD884F57B21D)
#define HPX_2F9C9286_A4A2_451C_BBD6_CD884F57B21D

#include <vector>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/constructor_argument.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

namespace hpx { namespace test { namespace server
{

struct HPX_COMPONENT_EXPORT managed_refcnt_checker
  : components::managed_component_base<managed_refcnt_checker>
{
  private:
    naming::id_type target_;
    std::vector<naming::id_type> references_;

  public:
    managed_refcnt_checker()
      : target_(naming::invalid_id)
      , references_()
    {}

    managed_refcnt_checker(
        components::constructor_argument const& target_
        )
      : target_(boost::get<naming::id_type>(target_))
      , references_()
    {}

    ~managed_refcnt_checker();

    void take_reference(
        naming::id_type const& gid
        )
    {
        references_.push_back(gid);
    }

    enum actions
    {
        action_take_reference
    };

    typedef hpx::actions::action1<
        // component
        managed_refcnt_checker
        // action code
      , action_take_reference
        // arguments
      , naming::id_type const&
        // method
      , &managed_refcnt_checker::take_reference
    > take_reference_action;
};

}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::test::server::managed_refcnt_checker::take_reference_action,
    managed_refcnt_checker_take_reference_action);

#endif // HPX_2F9C9286_A4A2_451C_BBD6_CD884F57B21D

