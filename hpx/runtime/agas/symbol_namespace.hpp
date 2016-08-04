////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7)
#define HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/agas_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/util/function.hpp>

#include <boost/cstdint.hpp>

#include <memory>
#include <string>
#include <vector>

namespace hpx { namespace agas
{

struct symbol_namespace
{
    // {{{ nested types
    typedef server::symbol_namespace server_type;

    typedef hpx::util::function<
        void(std::string const&, naming::gid_type const&)
    > iterate_names_function_type;
    // }}}

    static naming::gid_type get_service_instance(boost::uint32_t service_locality_id);

    static naming::gid_type get_service_instance(naming::gid_type const& dest,
        error_code& ec = throws);

    static naming::gid_type get_service_instance(naming::id_type const& dest)
    {
        return get_service_instance(dest.get_gid());
    }

    static bool is_service_instance(naming::gid_type const& gid);

    static bool is_service_instance(naming::id_type const& id)
    {
        return is_service_instance(id.get_gid());
    }

    static naming::id_type symbol_namespace_locality(std::string const& key);

    symbol_namespace();
    ~symbol_namespace();

    naming::address::address_type ptr() const;
    naming::address addr() const;
    naming::id_type gid() const;

    hpx::future<bool> bind_async(std::string key, naming::gid_type gid);
    bool bind(std::string key, naming::gid_type gid);

    hpx::future<naming::id_type> resolve_async(std::string key);
    naming::id_type resolve(std::string key);

    hpx::future<naming::id_type> unbind_async(std::string key);
    naming::id_type unbind(std::string key);

    hpx::future<bool> on_event(
        std::string const& name
      , bool call_for_past_events
      , hpx::id_type lco
        );

    void register_counter_types();
    void register_server_instance(boost::uint32_t locality_id);
    void unregister_server_instance(error_code& ec);

private:
    std::unique_ptr<server_type> server_;
};

}}

#endif // HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7

