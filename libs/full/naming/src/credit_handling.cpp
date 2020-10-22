//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/naming/credit_handling.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/server/destroy_component.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/split_gid.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/thread_support/unlock_guard.hpp>

#include <cstdint>
#include <functional>
#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
//
//          Here is how our distributed garbage collection works
//
// Each id_type instance - while always referring to some (possibly remote)
// entity - can either be 'managed' or 'unmanaged'. If an id_type instance is
// 'unmanaged' it does not perform any garbage collection. Otherwise (if it's
// 'managed'), all of its copies are globally tracked which allows to
// automatically delete the entity a particular id_type instance is referring
// to after the last reference to it goes out of scope.
//
// An id_type instance is essentially a shared_ptr<> maintaining two reference
// counts: a local reference count and a global one. The local reference count
// is incremented whenever the id_type instance is copied locally, and decremented
// whenever one of the local copies goes out of scope. At the point when the last
// local copy goes out of scope, it returns its current share of the global
// reference count back to AGAS. The share of the global reference count owned
// by all copies of an id_type instance on a single locality is called its
// credit. Credits are issued in chunks which allows to create a global copy
// of an id_type instance (like passing it to another locality) without needing
// to talk to AGAS to request a global reference count increment. The referenced
// entity is freed when the global reference count falls to zero.
//
// Any newly created object assumes an initial credit. This credit is not
// accounted for by AGAS as long as no global increment or decrement requests
// are received. It is important to understand that there is no way to distinguish
// whether an object has already been deleted (and therefore no entry exists in
// the table storing the global reference count for this object) or whether the
// object is still alive but no increment/decrement requests have been received
// by AGAS yet. While this is a pure optimization to avoid storing global
// reference counts for all objects, it has implications for the implemented
// garbage collection algorithms at large.
//
// As long as an id_type instance is not sent to another locality (a locality
// different from the initial locality creating the referenced entity), all
// lifetime management for this entity can be handled purely local without
// even talking to AGAS.
//
// Sending an id_type instance to another locality (which includes using an
// id_type as the destination for an action) splits the current credit into
// two parts. One part stays with the id_type on the sending locality, the
// other part is sent along to the destination locality where it turns into the
// global credit associated with the remote copy of the id_type. As stated
// above, this allows to avoid talking to AGAS for incrementing the global
// reference count as long as there is sufficient global credit left in order
// to be split.
//
// The current share of the global credit associated with an id_type instance
// is encoded in the bits 88..92 of the underlying gid_type (encoded as the
// logarithm to the base 2 of the credit value). Bit 94 is a flag which is set
// whenever the credit is valid. Bit 95 encodes whether the given id_type
// has been split at any time. This information is needed to be able to decide
// whether a garbage collection can be assumed to be a purely local operation.
// Bit 93 is used by the locking scheme for gid_types.
//
// Credit splitting is performed without any additional AGAS traffic as long as
// sufficient credit is available. If the credit of the id_type to be split is
// exhausted (reaches the value '1') it has to be replenished. This operation
// is performed synchronously. This is done to ensure that AGAS has accounted
// for the requested credit increase.
//
// Note that both the id_type instance staying behind and the one sent along
// are replenished before sending out the parcel at the sending locality.
//
///////////////////////////////////////////////////////////////////////////////

namespace hpx { namespace naming {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        void decrement_refcnt(gid_type* p)
        {
            // do nothing if it's too late in the game
            if (!get_runtime_ptr())
            {
                // delete local gid representation in any case
                delete p;
                return;
            }

            // Talk to AGAS only if this gid was split at some time in the past,
            // i.e. if a reference actually left the original locality.
            // Alternatively we need to go this way if the id has never been
            // resolved, which means we don't know anything about the component
            // type.
            naming::address addr;
            if (gid_was_split(*p) ||
                !naming::get_agas_client().resolve_cached(*p, addr))
            {
                // guard for wait_abort and other shutdown issues
                try
                {
                    // decrement global reference count for the given gid,
                    std::int64_t credits = detail::get_credit_from_gid(*p);
                    HPX_ASSERT(0 != credits);

                    if (get_runtime_ptr())    // -V547
                    {
                        // Fire-and-forget semantics.
                        error_code ec(lightweight);
                        agas::decref(*p, credits, ec);
                    }
                }
                catch (hpx::exception const& e)
                {
                    LTM_(error) << "Unhandled exception while executing "
                                   "decrement_refcnt:"
                                << e.what();
                }
            }
            else
            {
                // If the gid was not split at any point in time we can assume
                // that the referenced object is fully local.
                HPX_ASSERT(addr.type_ != components::component_invalid);

                // Third parameter is the count of how many components to destroy.
                // FIXME: The address should still be in the cache, but it could
                // be evicted. It would be nice to have a way to pass the address
                // directly to destroy_component.
                try
                {
                    components::server::destroy_component(*p, addr);
                }
                catch (hpx::exception const& e)
                {
                    // This request might come in too late and the thread manager
                    // was already stopped. We ignore the request if that's the
                    // case.
                    if (e.get_error() != invalid_status ||
                        !threads::threadmanager_is(hpx::state_stopping))
                    {
                        throw;
                    }
                }
            }
            delete p;    // delete local gid representation in any case
        }

        ///////////////////////////////////////////////////////////////////////
        hpx::future<gid_type> split_gid_if_needed(gid_type& gid)
        {
            std::unique_lock<gid_type::mutex_type> l(gid.get_mutex());
            return split_gid_if_needed_locked(l, gid);
        }

        gid_type postprocess_incref(gid_type& gid)
        {
            std::unique_lock<gid_type::mutex_type> l(gid.get_mutex());

            gid_type new_gid = gid;    // strips lock-bit
            HPX_ASSERT(new_gid != invalid_gid);

            // The old gid should have been marked as been split below
            HPX_ASSERT(gid_was_split(gid));

            // Fill the new gid with our new credit and mark it as being split
            naming::detail::set_credit_for_gid(
                new_gid, static_cast<std::int64_t>(HPX_GLOBALCREDIT_INITIAL));
            set_credit_split_mask_for_gid(new_gid);

            // Another concurrent split operation might have happened
            // concurrently, we need to add the new split credits to the old
            // and account for overflow.

            // Get the current credit for our gid. If no other concurrent
            // split has happened since we invoked incref below, the credit
            // of this gid is equal to 2, otherwise it is larger.
            std::int64_t src_credit = get_credit_from_gid(gid);
            HPX_ASSERT(src_credit >= 2);

            std::int64_t split_credit =
                static_cast<std::int64_t>(HPX_GLOBALCREDIT_INITIAL) - 2;
            std::int64_t new_credit = src_credit + split_credit;
            std::int64_t overflow_credit = new_credit -
                static_cast<std::int64_t>(HPX_GLOBALCREDIT_INITIAL);
            HPX_ASSERT(overflow_credit >= 0);

            new_credit =
                (std::min)(static_cast<std::int64_t>(HPX_GLOBALCREDIT_INITIAL),
                    new_credit);
            naming::detail::set_credit_for_gid(gid, new_credit);

            // Account for a possible overflow ...
            if (overflow_credit > 0)
            {
                HPX_ASSERT(overflow_credit <= HPX_GLOBALCREDIT_INITIAL - 1);
                l.unlock();

                // Note that this operation may be asynchronous
                agas::decref(new_gid, overflow_credit);
            }

            return new_gid;
        }

        hpx::future<gid_type> split_gid_if_needed_locked(
            std::unique_lock<gid_type::mutex_type>& l, gid_type& gid)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            if (naming::detail::has_credits(gid))
            {
                // The splitting is happening in two parts:
                // First get the current credit and split it:
                // Case 1: credit == 1 ==> we need to request new credit from
                //                         AGAS. This is happening asynchronously.
                // Case 2: credit != 1 ==> Just fill with new credit
                //
                // Scenario that might happen:
                // An id_type which needs to be split is being split concurrently
                // while we unlock the lock to ask for more credit:
                //     This might lead to an overflow in the credit mask and
                //     needs to be accounted for by sending a decref with the
                //     excessive credit.
                //
                // An early decref can't happen as the id_type with the new
                // credit is guaranteed to arrive only after we incremented the
                // credit successfully in agas.
                HPX_ASSERT(get_log2credit_from_gid(gid) > 0);
                std::int16_t src_log2credits = get_log2credit_from_gid(gid);

                // Credit exhaustion - we need to get more.
                if (src_log2credits == 1)
                {
                    // mark gid as being split
                    set_credit_split_mask_for_gid(gid);

                    l.unlock();

                    // We add HPX_GLOBALCREDIT_INITIAL credits for the new gid
                    // and HPX_GLOBALCREDIT_INITIAL - 2 for the old one.
                    std::int64_t new_credit = 2 *
                        (static_cast<std::int64_t>(HPX_GLOBALCREDIT_INITIAL) -
                            1);

                    naming::gid_type new_gid = gid;    // strips lock-bit
                    HPX_ASSERT(new_gid != invalid_gid);
                    return agas::incref(new_gid, new_credit)
                        .then(hpx::launch::sync,
                            hpx::util::bind(postprocess_incref, std::ref(gid)));
                }

                HPX_ASSERT(src_log2credits > 1);

                naming::gid_type new_gid = split_credits_for_gid_locked(l, gid);

                HPX_ASSERT(detail::has_credits(gid));
                HPX_ASSERT(detail::has_credits(new_gid));

                return hpx::make_ready_future(new_gid);
            }

            naming::gid_type new_gid = gid;    // strips lock-bit
            return hpx::make_ready_future(new_gid);
        }

        ///////////////////////////////////////////////////////////////////////
        gid_type move_gid(gid_type& gid)
        {
            std::unique_lock<gid_type::mutex_type> l(gid.get_mutex());
            return move_gid_locked(std::move(l), gid);
        }

        gid_type move_gid_locked(
            std::unique_lock<gid_type::mutex_type> l, gid_type& gid)    //-V813
        {
            HPX_ASSERT_OWNS_LOCK(l);

            naming::gid_type new_gid = gid;    // strips lock-bit

            if (naming::detail::has_credits(gid))
            {
                naming::detail::strip_credits_from_gid(gid);
            }

            return new_gid;
        }

        ///////////////////////////////////////////////////////////////////////
        gid_type split_credits_for_gid(gid_type& id)
        {
            std::unique_lock<gid_type::mutex_type> l(id.get_mutex());
            return split_credits_for_gid_locked(l, id);
        }

        gid_type split_credits_for_gid_locked(
            std::unique_lock<gid_type::mutex_type>& l, gid_type& id)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            std::uint16_t log2credits = get_log2credit_from_gid(id);
            HPX_ASSERT(log2credits > 0);

            gid_type newid = id;    // strips lock-bit

            set_log2credit_for_gid(id, log2credits - 1);
            set_credit_split_mask_for_gid(id);

            set_log2credit_for_gid(newid, log2credits - 1);
            set_credit_split_mask_for_gid(newid);

            return newid;
        }

        ///////////////////////////////////////////////////////////////////////
        std::int64_t replenish_credits(gid_type& gid)
        {
            std::unique_lock<gid_type::mutex_type> l(gid);
            return replenish_credits_locked(l, gid);
        }

        std::int64_t replenish_credits_locked(
            std::unique_lock<gid_type::mutex_type>& l, gid_type& gid)
        {
            std::int64_t added_credit = 0;

            HPX_ASSERT(0 == get_credit_from_gid(gid));

            added_credit = naming::detail::fill_credit_for_gid(gid);
            naming::detail::set_credit_split_mask_for_gid(gid);

            gid_type unlocked_gid = gid;    // strips lock-bit

            std::int64_t result = 0;
            {
                hpx::util::unlock_guard<std::unique_lock<gid_type::mutex_type>>
                    ul(l);
                result = agas::incref(launch::sync, unlocked_gid, added_credit);
            }

            return result;
        }

        std::int64_t add_credit_to_gid(gid_type& id, std::int64_t credits)
        {
            std::int64_t c = get_credit_from_gid(id);

            c += credits;
            set_credit_for_gid(id, c);

            return c;
        }

        std::int64_t remove_credit_from_gid(gid_type& id, std::int64_t debit)
        {
            std::int64_t c = get_credit_from_gid(id);
            HPX_ASSERT(c > debit);

            c -= debit;
            set_credit_for_gid(id, c);

            return c;
        }

        std::int64_t fill_credit_for_gid(gid_type& id, std::int64_t credits)
        {
            std::int64_t c = get_credit_from_gid(id);
            HPX_ASSERT(c <= credits);

            std::int64_t added = credits - c;
            set_credit_for_gid(id, credits);

            return added;
        }
    }    // namespace detail

    void decrement_refcnt(gid_type const& gid)
    {
        // We assume that the gid was split in the past
        HPX_ASSERT(detail::gid_was_split(gid));

        // decrement global reference count for the given gid,
        std::int64_t credits = detail::get_credit_from_gid(gid);
        HPX_ASSERT(0 != credits);

        // Fire-and-forget semantics.
        error_code ec(lightweight);
        agas::decref(gid, credits, ec);
    }
}}    // namespace hpx::naming
