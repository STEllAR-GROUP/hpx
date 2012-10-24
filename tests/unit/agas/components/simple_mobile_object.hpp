////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_1872ED70_555A_4F63_A8CE_113CFE76215C)
#define HPX_1872ED70_555A_4F63_A8CE_113CFE76215C

#include <hpx/async.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <tests/unit/agas/components/server/simple_mobile_object.hpp>

namespace hpx { namespace test
{

struct simple_mobile_object
  : components::client_base<
        simple_mobile_object
      , components::stub_base<server::simple_mobile_object>
    >
{
    typedef components::client_base<
        simple_mobile_object
      , components::stub_base<server::simple_mobile_object>
    > base_type;

  private:
    using base_type::create;

  public:
    typedef server::simple_mobile_object server_type;

    /// Create a new component on the target locality.
    explicit simple_mobile_object(
        naming::id_type const& locality
        )
    {
        this->base_type::create(locality);
    }

    boost::uint64_t get_lva()
    {
        typedef server_type::get_lva_action action_type;
        return hpx::async<action_type>(this->gid_).get();
    }
};

}}

#endif // HPX_1872ED70_555A_4F63_A8CE_113CFE76215C

