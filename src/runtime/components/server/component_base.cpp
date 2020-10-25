//  Copyright (c) 2015 Thomas Heller
//  Copyright (c) 2018-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_distributed/applier/applier.hpp>
#include <hpx/async_distributed/applier/bind_naming_wrappers.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming/credit_handling.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/runtime_fwd.hpp>

#include <cstdint>
#include <mutex>
#include <sstream>

namespace hpx { namespace components { namespace detail
{
    base_component::~base_component()
    {
        if (gid_)
        {
            error_code ec;
            agas::unbind(launch::sync, gid_, 1, ec);
        }
    }

    naming::gid_type base_component::get_base_gid_dynamic(
        naming::gid_type const& assign_gid, naming::address const& addr,
        naming::gid_type (*f)(naming::gid_type)) const
    {
        if (!gid_)
        {
            if (!assign_gid)
            {
                if (f != nullptr)
                {
                    gid_ = f(hpx::detail::get_next_id());
                }
                else
                {
                    gid_ = hpx::detail::get_next_id();
                }

                if (!applier::bind_gid_local(gid_, addr))
                {
                    std::ostringstream strm;
                    strm << "failed to bind id " << gid_
                            << "to locality: " << hpx::get_locality();

                    gid_ = naming::invalid_gid;    // invalidate GID

                    HPX_THROW_EXCEPTION(duplicate_component_address,
                        "component_base<Component>::get_base_gid",
                        strm.str());
                }
            }
            else
            {
                applier::applier& appl = hpx::applier::get_applier();
                gid_ = assign_gid;
                naming::detail::strip_credits_from_gid(gid_);

                if (!agas::bind(
                        launch::sync, gid_, addr, appl.get_locality_id()))
                {
                    std::ostringstream strm;
                    strm << "failed to rebind id " << gid_
                            << "to locality: " << hpx::get_locality();

                    gid_ = naming::invalid_gid;    // invalidate GID

                    HPX_THROW_EXCEPTION(duplicate_component_address,
                        "component_base<Component>::get_base_gid",
                        strm.str());
                }
            }
        }

        std::unique_lock<naming::gid_type::mutex_type> l(gid_.get_mutex());

        if (!naming::detail::has_credits(gid_))
        {
            naming::gid_type gid = gid_;
            return gid;
        }

        // on first invocation take all credits to avoid a self reference
        naming::gid_type gid = gid_;

        naming::detail::strip_credits_from_gid(
            const_cast<naming::gid_type&>(gid_));

        HPX_ASSERT(naming::detail::has_credits(gid));

        // We have to assume this credit was split as otherwise the gid
        // returned at this point will control the lifetime of the
        // component.
        naming::detail::set_credit_split_mask_for_gid(gid);
        return gid;
    }

    naming::gid_type base_component::get_base_gid(naming::address const& addr,
        naming::gid_type (*f)(naming::gid_type)) const
    {
        if (!gid_)
        {
            // generate purely local gid
            if (f != nullptr)
            {
                gid_ = f(naming::gid_type(addr.address_));
            }
            else
            {
                gid_ = naming::gid_type(addr.address_);
            }

            naming::detail::set_credit_for_gid(
                gid_, std::int64_t(HPX_GLOBALCREDIT_INITIAL));
            gid_ = naming::replace_component_type(gid_, addr.type_);
            gid_ = naming::replace_locality_id(gid_, agas::get_locality_id());

            // there is no need to explicitly bind this id in AGAS as the id
            // can be directly resolved to the address it contains.
        }

        std::unique_lock<naming::gid_type::mutex_type> l(gid_.get_mutex());

        if (!naming::detail::has_credits(gid_))
        {
            naming::gid_type gid = gid_;
            return gid;
        }

        // on first invocation take all credits to avoid a self reference
        naming::gid_type gid = gid_;

        naming::detail::strip_credits_from_gid(gid_);

        HPX_ASSERT(naming::detail::has_credits(gid));

        // We have to assume this credit was split as otherwise the gid
        // returned at this point will control the lifetime of the
        // component.
        naming::detail::set_credit_split_mask_for_gid(gid);
        return gid;
    }

    naming::id_type base_component::get_id(naming::gid_type gid) const
    {
        // all credits should have been taken already
        HPX_ASSERT(!naming::detail::has_credits(gid));

        // any (subsequent) invocation causes the credits to be replenished
        naming::detail::replenish_credits(gid);
        return naming::id_type(gid, naming::id_type::managed);
    }

    naming::id_type base_component::get_unmanaged_id(
        naming::gid_type const& gid) const
    {
        return naming::id_type(gid, naming::id_type::managed);
    }
}}}

