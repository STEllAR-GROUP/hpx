////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_5D993B14_5B65_4231_A84E_90AD1807EB8F)
#define HPX_5D993B14_5B65_4231_A84E_90AD1807EB8F

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/lcos/future.hpp>

#include <utility>
#include <vector>

namespace hpx { namespace agas { namespace stubs
{

struct HPX_EXPORT primary_namespace
{
    typedef server::primary_namespace server_type;
    typedef server::primary_namespace server_component_type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    static lcos::future<Result> service_async(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority = threads::thread_priority_default
        );

    /// Fire-and-forget semantics.
    ///
    /// \note This is placed out of line to avoid including applier headers.
    static void service_non_blocking(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority = threads::thread_priority_default
        );

    static void service_non_blocking(
        naming::id_type const& gid
      , request const& req
      , util::function_nonser<void(boost::system::error_code const&,
            parcelset::parcel const&)> const& f
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

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future<std::vector<response> > bulk_service_async(
        naming::id_type const& gid
      , std::vector<request> reqs
      , threads::thread_priority priority = threads::thread_priority_default
        );

    /// Fire-and-forget semantics.
    ///
    /// \note This is placed out of line to avoid including applier headers.
    static void bulk_service_non_blocking(
        naming::id_type const& gid
      , std::vector<request> reqs
      , threads::thread_priority priority = threads::thread_priority_default
        );

    static std::vector<response> bulk_service(
        naming::id_type const& gid
      , std::vector<request> reqs
      , threads::thread_priority priority = threads::thread_priority_default
      , error_code& ec = throws
        )
    {
        return bulk_service_async(gid, std::move(reqs), priority).get(ec);
    }

    static naming::gid_type get_service_instance(naming::gid_type const& dest)
    {
        boost::uint32_t service_locality_id = naming::get_locality_id_from_gid(dest);
        if (service_locality_id == naming::invalid_locality_id)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "primary_namespace::get_service_instance",
                boost::str(boost::format(
                        "can't retrieve a valid locality id from global address (%1%): "
                    ) % dest));
            return naming::gid_type();
        }
        naming::gid_type service(HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB);
        return naming::replace_locality_id(service, service_locality_id);
    }

    static naming::gid_type get_service_instance(naming::id_type const& dest)
    {
        return get_service_instance(dest.get_gid());
    }

    static bool is_service_instance(naming::gid_type const& gid)
    {
        return gid.get_lsb() == HPX_AGAS_PRIMARY_NS_LSB &&
            (gid.get_msb() & ~naming::gid_type::locality_id_mask) == HPX_AGAS_NS_MSB;
    }

    static bool is_service_instance(naming::id_type const& id)
    {
        return is_service_instance(id.get_gid());
    }
};

}}}

#endif // HPX_5D993B14_5B65_4231_A84E_90AD1807EB8F

