////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB)
#define HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

#include <boost/format.hpp>
#include <boost/utility/binary.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/runtime/actions/function.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/database/table.hpp>
#include <hpx/runtime/agas/namespace/response.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database, typename Protocol>
struct symbol_namespace :
  components::fixed_component_base<
    HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB, // constant GID
    symbol_namespace<Database, Protocol>
  >
{
    // {{{ nested types
    typedef typename traits::database::mutex_type<Database>::type
        database_mutex_type;

    typedef std::string symbol_type;

    typedef hpx::actions::function<
        void(symbol_type const&, naming::gid_type const&)
    > iterate_function_type;

    typedef response<Protocol> response_type;

    typedef table<Database, symbol_type, naming::gid_type>
        gid_table_type; 
    // }}} 
 
  private:
    database_mutex_type mutex_;
    gid_table_type gids_;
  
  public:
    symbol_namespace(
        std::string const& name = "root_symbol_namespace"
        )
      : mutex_()
      , gids_(std::string("hpx.agas.") + name + ".gid")
    {
        traits::initialize_mutex(mutex_);
    }
    
    response_type bind(
        symbol_type const& key
      , naming::gid_type const& gid
        )
    {
        return bind(key, gid, throws);
    }

    response_type bind(
        symbol_type const& key
      , naming::gid_type const& gid
      , error_code& ec
        )
    { // {{{ bind implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        // Always load the table once, as this operation might be slow for some
        // database backends.
        typename gid_table_type::map_type& gid_table = gids_.get();

        typename gid_table_type::map_type::iterator
            it = gid_table.find(key), end = gid_table.end();

        if (it != end)
        {
            naming::gid_type old = it->second;
            it->second = gid;

            LAGAS_(info) << (boost::format(
                "symbol_namespace::bind, key(%1%), gid(%2%), old_gid(%3%)")
                % key % gid % old);

            if (&ec != &throws)
                ec = make_success_code();

            return response_type(symbol_ns_bind, old);
        }

        if (HPX_UNLIKELY(!util::insert_checked(gid_table.insert(
                std::make_pair(key, gid)))))
        {
            HPX_THROWS_IF(ec, lock_error
              , "symbol_namespace::bind"
              , "GID table insertion failed due to a locking error or "
                "memory corruption");
            return response_type();
        }

        LAGAS_(info) << (boost::format(
            "symbol_namespace::bind, key(%1%), gid(%2%), old_gid(%3%)")
            % key % gid % gid);

        if (&ec != &throws)
            ec = make_success_code();

        return response_type(symbol_ns_bind, gid); 
    } // }}}

    response_type resolve(
        symbol_type const& key
        )
    {
        return resolve(key, throws);
    }

    response_type resolve(
        symbol_type const& key
      , error_code& ec
        )
    { // {{{ resolve implementation
        typename database_mutex_type::scoped_lock l(mutex_);

        typename gid_table_type::map_type& gid_table = gids_.get();

        typename gid_table_type::map_type::iterator
            it = gid_table.find(key), end = gid_table.end();

        if (it == end)
        {
            LAGAS_(info) << (boost::format(
                "symbol_namespace::resolve, key(%1%), response(no_success)")
                % key);

            if (&ec != &throws)
                ec = make_success_code();

            return response_type(symbol_ns_resolve
                               , naming::invalid_gid
                               , no_success);
        }

        LAGAS_(info) << (boost::format(
            "symbol_namespace::resolve, key(%1%), gid(%2%)")
            % key % it->second);

        if (&ec != &throws)
            ec = make_success_code();

        return response_type(symbol_ns_resolve, it->second);
    } // }}}  
    
    response_type unbind(
        symbol_type const& key
        )
    {
        return unbind(key, throws);
    }

    response_type unbind(
        symbol_type const& key
      , error_code& ec
        )
    { // {{{ unbind implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        
        typename gid_table_type::map_type& gid_table = gids_.get();

        typename gid_table_type::map_type::iterator
            it = gid_table.find(key), end = gid_table.end();

        if (it == end)
        {
            LAGAS_(info) << (boost::format(
                "symbol_namespace::unbind, key(%1%), response(no_success)")
                % key);

            if (&ec != &throws)
                ec = make_success_code();

            return response_type(symbol_ns_unbind, no_success);
        }

        gid_table.erase(it);

        LAGAS_(info) << (boost::format(
            "symbol_namespace::unbind, key(%1%)")
            % key);

        if (&ec != &throws)
            ec = make_success_code();

        return response_type(symbol_ns_unbind);
    } // }}} 

    response_type iterate(
        iterate_function_type const& f
        )
    {
        return iterate(f, throws);
    }

    response_type iterate(
        iterate_function_type const& f
      , error_code& ec
        )
    { // {{{ iterate implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        
        typename gid_table_type::map_type& gid_table = gids_.get();

        for (typename gid_table_type::map_type::iterator it = gid_table.begin(),
                                                         end = gid_table.end();
             it != end; ++it)
        {
            f(it->first, it->second);
        }

        LAGAS_(info) << "symbol_namespace::iterate";

        if (&ec != &throws)
            ec = make_success_code();

        return response_type(symbol_ns_iterate);
    } // }}} 

    enum actions
    { // {{{ action enum
        namespace_bind    = BOOST_BINARY_U(0010000),
        namespace_resolve = BOOST_BINARY_U(0010001),
        namespace_unbind  = BOOST_BINARY_U(0010010),
        namespace_iterate = BOOST_BINARY_U(0010011)
    }; // }}}
    
    typedef hpx::actions::result_action2<
        symbol_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_bind,
        /* arguments */   symbol_type const&, naming::gid_type const&,
        &symbol_namespace<Database, Protocol>::bind
      , threads::thread_priority_critical
    > bind_action;
    
    typedef hpx::actions::result_action1<
        symbol_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_resolve,
        /* arguments */   symbol_type const&,
        &symbol_namespace<Database, Protocol>::resolve
      , threads::thread_priority_critical
    > resolve_action;
    
    typedef hpx::actions::result_action1<
        symbol_namespace<Database, Protocol>,
        /* retrun type */ response_type,
        /* enum value */  namespace_unbind,
        /* arguments */   symbol_type const&,
        &symbol_namespace<Database, Protocol>::unbind
      , threads::thread_priority_critical
    > unbind_action;

    typedef hpx::actions::result_action1<
        symbol_namespace<Database, Protocol>,
        /* retrun type */ response_type,
        /* enum value */  namespace_iterate,
        /* arguments */   iterate_function_type const&,
        &symbol_namespace<Database, Protocol>::iterate
      , threads::thread_priority_critical
    > iterate_action;
};

}}}

#endif // HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

