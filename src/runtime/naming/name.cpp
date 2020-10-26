//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/lcos_local/detail/preprocess_future.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/memory/serialization/intrusive_ptr.hpp>
#include <hpx/modules/checkpoint_base.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/server/destroy_component.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/split_gid.hpp>
#include <hpx/runtime/serialization/detail/preprocess_gid_types.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>
#include <hpx/state.hpp>
#include <hpx/thread_support/assert_owns_lock.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/util/ios_flags_saver.hpp>

#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace detail
{
    struct gid_serialization_data;
}}}

HPX_IS_BITWISE_SERIALIZABLE(hpx::naming::detail::gid_serialization_data)

namespace hpx { namespace naming
{
    ///////////////////////////////////////////////////////////////////////////
    util::internal_allocator<detail::id_type_impl> detail::id_type_impl::alloc_;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // custom deleter for managed gid_types, will be called when the last
        // copy of the corresponding naming::id_type goes out of scope
        void gid_managed_deleter(id_type_impl* p)
        {
            // a credit of zero means the component is not (globally) reference
            // counted
            if (detail::has_credits(*p))
            {
                // execute the deleter directly
                detail::decrement_refcnt(p);
            }
            else
            {
                // delete local gid representation if needed
                delete p;
            }
        }

        // custom deleter for unmanaged gid_types, will be called when the last
        // copy of the corresponding naming::id_type goes out of scope
        void gid_unmanaged_deleter(id_type_impl* p)
        {
            delete p;   // delete local gid representation only
        }

        ///////////////////////////////////////////////////////////////////////
        id_type_impl::deleter_type id_type_impl::get_deleter(
            id_type_management t) noexcept
        {
            switch (t)
            {
            case unmanaged:
                return &detail::gid_unmanaged_deleter;

            case managed:
            case managed_move_credit:
                return &detail::gid_managed_deleter;

            default:
                HPX_ASSERT(false);          // invalid management type
                return &detail::gid_unmanaged_deleter;
            }
            return nullptr;
        }

        ///////////////////////////////////////////////////////////////////////
        // prepare the given id, note: this function modifies the passed id
        void handle_credit_splitting(serialization::output_archive& ar,
            id_type_impl& gid,
            serialization::detail::preprocess_gid_types& split_gids)
        {
            auto& handle_futures =
                ar.get_extra_data<serialization::detail::preprocess_futures>();

            // avoid races between the split-handling and the future
            // preprocessing
            handle_futures.increment_future_count();

            auto f = split_gid_if_needed(gid).then(hpx::launch::sync,
                [&split_gids, &handle_futures, &gid](
                    hpx::future<gid_type>&& gid_future) {
                        HPX_UNUSED(handle_futures);
                        HPX_ASSERT(handle_futures.has_futures());
                        split_gids.add_gid(gid, gid_future.get());
                });

            handle_futures.await_future(
                *traits::future_access<decltype(f)>::get_shared_state(f),
                false);
        }

        void id_type_impl::preprocess_gid(serialization::output_archive& ar) const
        {
            // unmanaged gids do not require any special handling
            // check-pointing does not require any special handling here neither
            if (unmanaged == type_ || managed_move_credit == type_)
            {
                return;
            }

            // we should not call this function during check-pointing operations
            if (ar.try_get_extra_data<util::checkpointing_tag>() != nullptr)
            {
                // this is a check-pointing operation, we do not support this
                HPX_THROW_EXCEPTION(invalid_status,
                    "id_type_impl::preprocess_gid",
                    "can't check-point managed id_type's, use a component "
                    "client instead");
            }

            auto& split_gids = ar.get_extra_data<
                serialization::detail::preprocess_gid_types>();

            if (split_gids.has_gid(*this))
            {
                // the gid has been split already and we don't need to do
                // anything further
                return;
            }

            HPX_ASSERT(has_credits(*this));

            // Request new credits from AGAS if needed (i.e. the remainder
            // of the credit splitting is equal to one).
            HPX_ASSERT(managed == type_);

            handle_credit_splitting(
                ar, const_cast<id_type_impl&>(*this), split_gids);
        }

        ///////////////////////////////////////////////////////////////////////
        struct gid_serialization_data
        {
            gid_type gid_;
            detail::id_type_management type_;

            template <typename Archive>
            void serialize(Archive& ar, unsigned)
            {
                ar & gid_ & type_;
            }
        };

        // serialization
        void id_type_impl::save(serialization::output_archive& ar, unsigned) const
        {
            // Avoid performing side effects if the archive is not saving the
            // data.
            if (ar.is_preprocessing())
            {
                preprocess_gid(ar);
                gid_serialization_data data { *this, type_ };
                ar << data;
                return;
            }

            id_type_management type = type_;

            if (unmanaged != type_ &&
                ar.try_get_extra_data<util::checkpointing_tag>() != nullptr)
            {
                // this is a check-pointing operation, we do not support this
                HPX_THROW_EXCEPTION(invalid_status,
                    "id_type_impl::save",
                    "can't check-point managed id_type's, use a component "
                    "client instead");
            }

            gid_type new_gid;
            if (unmanaged == type_)
            {
                new_gid = *this;
            }
            else if (managed_move_credit == type_)
            {
                // all credits will be moved to the returned gid
                new_gid = move_gid(const_cast<id_type_impl&>(*this));
                type = managed;
            }
            else
            {
                auto& split_gids = ar.get_extra_data<
                    serialization::detail::preprocess_gid_types>();

                new_gid = split_gids.get_new_gid(*this);
                HPX_ASSERT(new_gid != invalid_gid);
            }

#if defined(HPX_DEBUG)
            auto* split_gids = ar.try_get_extra_data<
                serialization::detail::preprocess_gid_types>();
            HPX_ASSERT(!split_gids || !split_gids->has_gid(*this));
#endif

            gid_serialization_data data{new_gid, type};
            ar << data;
        }

        void id_type_impl::load(serialization::input_archive& ar, unsigned)
        {
            gid_serialization_data data;
            ar >> data;

            static_cast<gid_type&>(*this) = data.gid_;
            type_ = static_cast<id_type_management>(data.type_);

            if (detail::unmanaged != type_ && detail::managed != type_)
            {
                HPX_THROW_EXCEPTION(version_too_new, "id_type::load",
                    "trying to load id_type with unknown deleter");
            }
        }

        /// support functions for hpx::intrusive_ptr
        void intrusive_ptr_add_ref(id_type_impl* p)
        {
            ++p->count_;
        }

        void intrusive_ptr_release(id_type_impl* p)
        {
            if (0 == --p->count_)
                id_type_impl::get_deleter(p->get_management_type())(p);
        }
    }   // detail

    std::ostream& operator<<(std::ostream& os, id_type const& id)
    {
        if (!id)
        {
            os << "{invalid}";
        }
        else
        {
            os << id.get_gid();
        }
        return os;
    }

    ///////////////////////////////////////////////////////////////////////////
    void id_type::save(
        serialization::output_archive& ar,
        const unsigned int version) const
    {
        // We serialize the intrusive ptr and use pointer tracking here. This
        // avoids multiple credit splitting if we need multiple future await
        // passes (they all work on the same archive).
        ar << gid_;
    }

    void id_type::load(
        serialization::input_archive& ar,
        const unsigned int version)
    {
        ar >> gid_;
    }

    ///////////////////////////////////////////////////////////////////////////
    char const* const management_type_names[] =
    {
        "unknown_deleter",      // -1
        "unmanaged",            // 0
        "managed",              // 1
        "managed_move_credit"   // 2
    };

    char const* get_management_type_name(id_type::management_type m)
    {
        if (m < id_type::unknown_deleter || m > id_type::managed_move_credit)
            return "invalid";
        return management_type_names[m + 1];
    }
}}

namespace hpx
{
    naming::id_type get_colocation_id(launch::sync_policy,
        naming::id_type const& id, error_code& ec)
    {
        return agas::get_colocation_id(launch::sync, id, ec);
    }

    lcos::future<naming::id_type> get_colocation_id(naming::id_type const& id)
    {
        return agas::get_colocation_id(id);
    }
}

