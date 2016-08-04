////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6)
#define HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

#include <hpx/config.hpp>
#include <hpx/runtime/agas_fwd.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/agas/gva.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/util/tuple.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace hpx { namespace agas
{

struct HPX_EXPORT primary_namespace
{
    typedef hpx::util::tuple<naming::gid_type, gva, naming::gid_type>
        resolved_type;

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

    primary_namespace();
    ~primary_namespace();

    naming::address::address_type ptr() const;
    naming::address addr() const;
    naming::id_type gid() const;

    future<std::pair<naming::id_type, naming::address>>
    begin_migration(naming::gid_type id);
    future<bool> end_migration(naming::gid_type id);

    bool bind_gid(gva g, naming::gid_type id, naming::gid_type locality);
    future<bool> bind_gid_async(gva g, naming::gid_type id, naming::gid_type locality);

    void route(parcelset::parcel && p,
        util::function_nonser<void(boost::system::error_code const&,
        parcelset::parcel const&)> && f);

    resolved_type resolve_gid(naming::gid_type id);
    future<resolved_type> resolve_full(naming::gid_type id);

    future<id_type> colocate(naming::gid_type id);

    naming::address unbind_gid(boost::uint64_t count, naming::gid_type id);
    future<naming::address>
    unbind_gid_async(boost::uint64_t count, naming::gid_type id);

    future<boost::int64_t> increment_credit(
        boost::int64_t credits
      , naming::gid_type lower
      , naming::gid_type upper
        );

    std::pair<naming::gid_type, naming::gid_type> allocate(boost::uint64_t count);

    void set_local_locality(naming::gid_type const& g);

    void register_counter_types();
    void register_server_instance(boost::uint32_t locality_id);
    void unregister_server_instance(error_code& ec);
private:
    std::unique_ptr<server::primary_namespace> server_;
};

}}

#endif // HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

