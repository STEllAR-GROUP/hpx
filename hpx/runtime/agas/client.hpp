////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_821AB60E_D2F1_4228_A10E_CD60CC0DB1DA)
#define HPX_821AB60E_D2F1_4228_A10E_CD60CC0DB1DA

#include <hpx/runtime/agas/namespaces.hpp>
#include <hpx/runtime/agas/refcnt_service.hpp>

namespace hpx { namespace agas 
{

template <typename Protocol>
struct agas_client
{
    typedef typename hpx::components::agas::symbol_namespace<Protocol>::type
        symbol_namespace_type;
    typedef typename hpx::components::agas::factory_namespace<Protocol>::type
        factory_namespace_type;
    typedef typename hpx::components::agas::locality_namespace<Protocol>::type
        locality_namespace_type;
    typedef typename hpx::components::agas::primary_namespace<Protocol>::type
        primary_namespace_type;

  private:
    hpx::components::agas::refcnt_service refcnt_service_;
    symbol_namespace_type symbol_ns_;
    factory_namespace_type factory_ns_;
    locality_namespace_type locality_ns_;
    primary_namespace_type primary_ns_;

  public:
    explicit agas_client(naming::id_type const& refcnt_service,
                         naming::id_type const& symbol_ns,
                         naming::id_type const& factory_ns,
                         naming::id_type const& locality_ns,
                         naming::id_type const& primary_ns):
        refcnt_service_(refcnt_service),
        symbol_ns_(symbol_ns),
        factory_ns_(factory_ns),
        locality_ns_(locality_ns),
        primary_ns_(primary_ns) {} 
};

}}

#endif // HPX_821AB60E_D2F1_4228_A10E_CD60CC0DB1DA

