//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/actions/continuation.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions/transfer_continuation_action.hpp>
#include <hpx/actions_base/traits/action_priority.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming/credit_handling.hpp>
#include <hpx/runtime/trigger_lco.hpp>

#include <exception>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions {

    continuation::continuation() = default;

    continuation::continuation(naming::id_type const& id)
      : id_(id)
    {
        // Try to resolve the address locally ...
        if (id_ && !agas::is_local_address_cached(id_, addr_))
        {
            addr_ = naming::address();
        }
    }

    continuation::continuation(naming::id_type&& id)
      : id_(std::move(id))
    {
        // Try to resolve the address locally ...
        if (id_ && !agas::is_local_address_cached(id_, addr_))
        {
            addr_ = naming::address();
        }
    }

    continuation::continuation(
        naming::id_type const& id, naming::address&& addr)
      : id_(id)
      , addr_(std::move(addr))
    {
    }

    continuation::continuation(
        naming::id_type&& id, naming::address&& addr) noexcept
      : id_(std::move(id))
      , addr_(std::move(addr))
    {
    }

    continuation::continuation(continuation&& o) noexcept = default;

    continuation& continuation::operator=(continuation&& o) noexcept = default;

    ///////////////////////////////////////////////////////////////////////////
    void continuation::trigger_error(std::exception_ptr const& e)
    {
        if (!id_)
        {
            HPX_THROW_EXCEPTION(invalid_status, "continuation::trigger_error",
                "attempt to trigger invalid LCO (the id is invalid)");
            return;
        }

        LLCO_(info) << "continuation::trigger_error(" << id_ << ")";
        set_lco_error(id_, this->get_addr(), e);
    }

    void continuation::trigger_error(std::exception_ptr&& e)    //-V659
    {
        if (!id_)
        {
            HPX_THROW_EXCEPTION(invalid_status, "continuation::trigger_error",
                "attempt to trigger invalid LCO (the id is invalid)");
            return;
        }

        LLCO_(info) << "continuation::trigger_error(" << id_ << ")";
        set_lco_error(id_, this->get_addr(), std::move(e));
    }

    void continuation::serialize(
        hpx::serialization::input_archive& ar, unsigned)
    {
        // clang-format off
        ar & id_ & addr_;
        // clang-format on
    }

    void continuation::serialize(
        hpx::serialization::output_archive& ar, unsigned)
    {
        // clang-format off
        ar & id_ & addr_;
        // clang-format on
    }

    ///////////////////////////////////////////////////////////////////////////
    void typed_continuation<void, util::unused_type>::serialize(
        hpx::serialization::input_archive& ar, unsigned)
    {
        // clang-format off
        // serialize base class
        ar & hpx::serialization::base_object<continuation>(*this);
        ar & f_;
        // clang-format on
    }

    void typed_continuation<void, util::unused_type>::serialize(
        hpx::serialization::output_archive& ar, unsigned)
    {
        // clang-format off
        // serialize base class
        ar & hpx::serialization::base_object<continuation>(*this);
        ar & f_;
        // clang-format on
    }

    void typed_continuation<void, util::unused_type>::trigger()
    {
        LLCO_(info) << "typed_continuation<void>::trigger(" << this->get_id()
                    << ")";

        if (f_.empty())
        {
            if (!this->get_id())
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "typed_continuation<void>::trigger",
                    "attempt to trigger invalid LCO (the id is invalid)");
                return;
            }
            trigger_lco_event(this->get_id(), this->get_addr());
        }
        else
        {
            f_(this->get_id());
        }
    }
}}    // namespace hpx::actions
