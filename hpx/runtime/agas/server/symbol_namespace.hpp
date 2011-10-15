////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB)
#define HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

#include <map>

#include <boost/format.hpp>
#include <boost/utility/binary.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/function.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/lcos/local_mutex.hpp>

namespace hpx { namespace agas
{

HPX_EXPORT naming::gid_type bootstrap_symbol_namespace_gid();
HPX_EXPORT naming::id_type bootstrap_symbol_namespace_id();

namespace server
{

struct symbol_namespace :
    components::fixed_component_base<
        HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB, // constant GID
        symbol_namespace
    >
{
    // {{{ nested types
    typedef lcos::local_mutex mutex_type;

    typedef hpx::actions::function<
        void(std::string const&, naming::gid_type const&)
    > iterate_names_function_type;

    typedef std::map<std::string, naming::gid_type> gid_table_type;
    // }}}

  private:
    mutex_type mutex_;
    gid_table_type gids_;

  public:
    symbol_namespace()
      : mutex_()
      , gids_()
    {}

    response service(
        request const& req
        )
    {
        return service(req, throws);
    }

    response service(
        request const& req
      , error_code& ec
        );

    response bind(
        request const& req
      , error_code& ec = throws
        );

    response resolve(
        request const& req
      , error_code& ec = throws
        );

    response unbind(
        request const& req
      , error_code& ec = throws
        );

    response iterate(
        request const& req
      , error_code& ec = throws
        );

    enum actions
    { // {{{ action enum
        // Actual actions
        namespace_service = BOOST_BINARY_U(0010000)

        // Pseudo-actions
      , namespace_bind    = BOOST_BINARY_U(0010001)
      , namespace_resolve = BOOST_BINARY_U(0010010)
      , namespace_unbind  = BOOST_BINARY_U(0010011)
      , namespace_iterate = BOOST_BINARY_U(0010100)
    }; // }}}

    typedef hpx::actions::result_action1<
        symbol_namespace
      , /* return type */ response
      , /* enum value */  namespace_service
      , /* arguments */   request const&
      , &symbol_namespace::service
      , threads::thread_priority_critical
    > service_action;
};

}}}

#endif // HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

