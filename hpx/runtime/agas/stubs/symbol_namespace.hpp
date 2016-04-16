////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_28443929_CB68_43ED_B134_F60602A344DD)
#define HPX_28443929_CB68_43ED_B134_F60602A344DD

#include <hpx/config.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime/agas/server/symbol_namespace.hpp>
#include <hpx/util/jenkins_hash.hpp>

#include <string>
#include <vector>

namespace hpx { namespace agas { namespace stubs
{

struct HPX_EXPORT symbol_namespace
{
    typedef server::symbol_namespace server_type;
    typedef server::symbol_namespace server_component_type;

    typedef server_type::iterate_names_function_type
        iterate_names_function_type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    static lcos::future<Result> service_async(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority = threads::thread_priority_default
        );

    template <typename Result>
    static lcos::future<Result> service_async(
        std::string const& key
      , request const& req
      , threads::thread_priority priority = threads::thread_priority_default
        )
    {
        return service_async<Result>(symbol_namespace_locality(key), req, priority);
    }

    /// Fire-and-forget semantics.
    ///
    /// \note This is placed out of line to avoid including applier headers.
    static void service_non_blocking(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority = threads::thread_priority_default
        );

    static response service(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority = threads::thread_priority_default
      , error_code& ec = throws
        )
    {
        return service_async<response>(gid, req, priority).get(ec);
    }

    static response service(
        std::string const& key
      , request const& req
      , threads::thread_priority priority = threads::thread_priority_default
      , error_code& ec = throws
        )
    {
        return service_async<response>(key, req, priority).get(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future<std::vector<response> > bulk_service_async(
        naming::id_type const& gid
      , std::vector<request> const& reqs
      , threads::thread_priority priority = threads::thread_priority_default
        );

    /// Fire-and-forget semantics.
    ///
    /// \note This is placed out of line to avoid including applier headers.
    static void bulk_service_non_blocking(
        naming::id_type const& gid
      , std::vector<request> const& reqs
      , threads::thread_priority priority = threads::thread_priority_default
        );

    static std::vector<response> bulk_service(
        naming::id_type const& gid
      , std::vector<request> const& reqs
      , threads::thread_priority priority = threads::thread_priority_default
      , error_code& ec = throws
        )
    {
        return bulk_service_async(gid, reqs, priority).get(ec);
    }

    static naming::gid_type get_service_instance(boost::uint32_t service_locality_id)
    {
        naming::gid_type service(HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB);
        return naming::replace_locality_id(service, service_locality_id);
    }

    static naming::gid_type get_service_instance(naming::gid_type const& dest,
        error_code& ec = throws)
    {
        boost::uint32_t service_locality_id = naming::get_locality_id_from_gid(dest);
        if (service_locality_id == naming::invalid_locality_id)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "symbol_namespace::get_service_instance",
                boost::str(boost::format(
                        "can't retrieve a valid locality id from global address (%1%): "
                    ) % dest));
            return naming::gid_type();
        }
        return get_service_instance(service_locality_id);
    }

    static naming::gid_type get_service_instance(naming::id_type const& dest)
    {
        return get_service_instance(dest.get_gid());
    }

    static bool is_service_instance(naming::gid_type const& gid)
    {
        return gid.get_lsb() == HPX_AGAS_SYMBOL_NS_LSB &&
            (gid.get_msb() & ~naming::gid_type::locality_id_mask) == HPX_AGAS_NS_MSB;
    }

    static bool is_service_instance(naming::id_type const& id)
    {
        return is_service_instance(id.get_gid());
    }

    static naming::id_type symbol_namespace_locality(std::string const& key)
    {
        boost::uint32_t hash_value = 0;
        if (key.size() < 2 || key[1] != '0' || key[0] != '/')
        {
            // keys starting with '/0' have to go to node 0
            util::jenkins_hash hash;
            hash_value = hash(key) % get_initial_num_localities();
        }
        return naming::id_type(get_service_instance(hash_value),
            naming::id_type::unmanaged);
    }
};

}}}

#endif // HPX_28443929_CB68_43ED_B134_F60602A344DD

