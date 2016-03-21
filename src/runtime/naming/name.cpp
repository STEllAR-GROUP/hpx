//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/exception.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/naming/split_gid.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/intrusive_ptr.hpp>
#include <hpx/util/assert_owns_lock.hpp>
#include <hpx/util/assert.hpp>

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <hpx/traits/is_bitwise_serializable.hpp>

#include <boost/mpl/bool.hpp>

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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace detail
{
    struct gid_serialization_data;
}}}

namespace hpx { namespace traits
{
    template <>
    struct is_bitwise_serializable<
            hpx::naming::detail::gid_serialization_data>
       : boost::mpl::true_
    {};
}}

namespace hpx { namespace naming
{
    namespace detail
    {
        void decrement_refcnt(detail::id_type_impl* p)
        {
            // do nothing if it's too late in the game
            if (!get_runtime_ptr())
            {
                delete p;   // delete local gid representation in any case
                return;
            }

            // Talk to AGAS only if this gid was split at some time in the past,
            // i.e. if a reference actually left the original locality.
            // Alternatively we need to go this way if the id has never been
            // resolved, which means we don't know anything about the component
            // type.
            naming::address addr;
            if ((gid_was_split(*p) ||
                !naming::get_agas_client().resolve_cached(*p, addr)))
            {
                // guard for wait_abort and other shutdown issues
                try {
                    // decrement global reference count for the given gid,
                    boost::int64_t credits = detail::get_credit_from_gid(*p);
                    HPX_ASSERT(0 != credits);

                    if (get_runtime_ptr())
                    {
                        // Fire-and-forget semantics.
                        error_code ec(lightweight);
                        agas::decref(*p, credits, ec);
                    }
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing decrement_refcnt:"
                        << e.what();
                }
            }
            else {
                // If the gid was not split at any point in time we can assume
                // that the referenced object is fully local.
                HPX_ASSERT(addr.type_ != components::component_invalid);

                // Third parameter is the count of how many components to destroy.
                // FIXME: The address should still be in the cache, but it could
                // be evicted. It would be nice to have a way to pass the address
                // directly to free_component_sync.
                try {
                    using components::stubs::runtime_support;
                    agas::gva g (addr.locality_, addr.type_, 1, addr.address_);
                    runtime_support::free_component_sync(g, *p);
                }
                catch (hpx::exception const& e) {
                    // This request might come in too late and the thread manager
                    // was already stopped. We ignore the request if that's the
                    // case.
                    if (e.get_error() != invalid_status) {
                        throw;      // rethrow if not invalid_status
                    }
                    else if (!threads::threadmanager_is(hpx::state_stopping)) {
                        throw;      // rethrow if not stopping
                    }
                }
            }
            delete p;   // delete local gid representation in any case
        }

        // custom deleter for managed gid_types, will be called when the last
        // copy of the corresponding naming::id_type goes out of scope
        void gid_managed_deleter (id_type_impl* p)
        {
            // a credit of zero means the component is not (globally) reference
            // counted
            if (detail::has_credits(*p)) {
                // execute the deleter directly
                decrement_refcnt(p);
            }
            else {
                delete p;   // delete local gid representation if needed
            }
        }

        // custom deleter for unmanaged gid_types, will be called when the last
        // copy of the corresponding naming::id_type goes out of scope
        void gid_unmanaged_deleter (id_type_impl* p)
        {
            delete p;   // delete local gid representation only
        }

        ///////////////////////////////////////////////////////////////////////
        id_type_impl::deleter_type id_type_impl::get_deleter(id_type_management t)
        {
            switch (t) {
            case unmanaged:
                return &detail::gid_unmanaged_deleter;

            case managed:
            case managed_move_credit:
                return &detail::gid_managed_deleter;

            default:
                HPX_ASSERT(false);          // invalid management type
                return &detail::gid_unmanaged_deleter;
            }
            return 0;
        }

        ///////////////////////////////////////////////////////////////////////
        // prepare the given id, note: this function modifies the passed id
        void id_type_impl::preprocess_gid(serialization::output_archive& ar) const
        {
            typedef gid_type::mutex_type::scoped_lock scoped_lock;
            // unmanaged gids do not require any special handling
            if (unmanaged == type_)
            {
                return;
            }

            HPX_ASSERT(has_credits(*this));

            // Request new credits from AGAS if needed (i.e. the remainder
            // of the credit splitting is equal to one).
            if (managed == type_)
            {
                ar.await_future(
                    split_gid_if_needed(const_cast<id_type_impl&>(*this)).then(
                        [&ar, this](hpx::future<gid_type> && gid_future)
                        {
                            ar.add_gid(*this, gid_future.get());
                        }
                    )
                );
                return;
            }
        }

        ///////////////////////////////////////////////////////////////////////
        hpx::future<gid_type> split_gid_if_needed(gid_type& gid)
        {
            typedef gid_type::mutex_type::scoped_lock scoped_lock;
            scoped_lock l(gid.get_mutex());
            return split_gid_if_needed_locked(l, gid);
        }

        gid_type postprocess_incref(gid_type &gid)
        {
            typedef gid_type::mutex_type::scoped_lock scoped_lock;
            scoped_lock l(gid.get_mutex());

            gid_type new_gid = gid;             // strips lock-bit
            HPX_ASSERT(new_gid != invalid_gid);

            // The old gid should have been marked as been split below
            HPX_ASSERT(gid_was_split(gid));

            // Fill the new gid with our new credit and mark it as being split
            naming::detail::set_credit_for_gid(new_gid, HPX_GLOBALCREDIT_INITIAL);
            set_credit_split_mask_for_gid(new_gid);

            // Another concurrent split operation might have happened
            // concurrently, we need to add the new split credits to the old
            // and account for overflow.

            // Get the current credit for our gid. If no other concurrent
            // split has happened since we invoked incref below, the credit
            // of this gid is equal to 2, otherwise it is larger.
            boost::int64_t src_credit = get_credit_from_gid(gid);
            HPX_ASSERT(src_credit >= 2);

            boost::int64_t split_credit = HPX_GLOBALCREDIT_INITIAL - 2;
            boost::int64_t new_credit = src_credit + split_credit;
            boost::int64_t overflow_credit = new_credit - HPX_GLOBALCREDIT_INITIAL;

            new_credit
                = (std::min)(
                    static_cast<boost::int64_t>(HPX_GLOBALCREDIT_INITIAL),
                    new_credit);
            naming::detail::set_credit_for_gid(gid, new_credit);

            // Account for a possible overflow ...
            if(overflow_credit > 0)
            {
                HPX_ASSERT(overflow_credit <= HPX_GLOBALCREDIT_INITIAL-1);
                l.unlock();

                // Note that this operation may be asynchronous
                agas::decref(new_gid, overflow_credit);
            }

            return new_gid;
        }

        hpx::future<gid_type> split_gid_if_needed_locked(
            gid_type::mutex_type::scoped_lock &l, gid_type& gid)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            typedef gid_type::mutex_type::scoped_lock scoped_lock;

            if (naming::detail::has_credits(gid))
            {
                // The splitting is happening in two parts:
                // First get the current credit and split it:
                // Case 1: credit == 1 ==> we need to request new credit from
                //                         AGAS. This is happening synchronously.
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
                boost::int16_t src_log2credits = get_log2credit_from_gid(gid);

                // Credit exhaustion - we need to get more.
                if(src_log2credits == 1)
                {
                    // mark gid as being split
                    set_credit_split_mask_for_gid(gid);

                    l.unlock();

                    // We add HPX_GLOBALCREDIT_INITIAL credits for the new gid
                    // and HPX_GLOBALCREDIT_INITIAL - 2 for the old one.
                    boost::int64_t new_credit = (HPX_GLOBALCREDIT_INITIAL - 1) * 2;

                    naming::gid_type new_gid = gid;     // strips lock-bit
                    HPX_ASSERT(new_gid != invalid_gid);
                    return agas::incref_async(new_gid, new_credit)
                        .then(
                            hpx::util::bind(postprocess_incref, boost::ref(gid))
                        );
                }

                HPX_ASSERT(src_log2credits > 1);

                naming::gid_type new_gid = split_credits_for_gid_locked(l, gid);

                HPX_ASSERT(detail::has_credits(gid));
                HPX_ASSERT(detail::has_credits(new_gid));

                return hpx::make_ready_future(new_gid);
            }

            naming::gid_type new_gid = gid; // strips lock-bit
            return hpx::make_ready_future(new_gid);
        }

        ///////////////////////////////////////////////////////////////////////
        gid_type move_gid(gid_type& gid)
        {
            gid_type::mutex_type::scoped_lock l(gid.get_mutex());
            return move_gid_locked(std::move(l), gid);
        }

        gid_type move_gid_locked(
            gid_type::mutex_type::scoped_lock l, gid_type& gid)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            naming::gid_type new_gid = gid;        // strips lock-bit

            if (naming::detail::has_credits(gid))
            {
                naming::detail::strip_credits_from_gid(gid);
            }

            return new_gid;
        }

        ///////////////////////////////////////////////////////////////////////
        gid_type split_credits_for_gid(gid_type& id)
        {
            typedef gid_type::mutex_type::scoped_lock scoped_lock;
            scoped_lock l(id.get_mutex());
            return split_credits_for_gid_locked(l, id);
        }

        gid_type split_credits_for_gid_locked(
            gid_type::mutex_type::scoped_lock& l, gid_type& id)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            boost::uint16_t log2credits = get_log2credit_from_gid(id);
            HPX_ASSERT(log2credits > 0);

            gid_type newid = id;            // strips lock-bit

            set_log2credit_for_gid(id, log2credits-1);
            set_credit_split_mask_for_gid(id);

            set_log2credit_for_gid(newid, log2credits-1);
            set_credit_split_mask_for_gid(newid);

            return newid;
        }

        ///////////////////////////////////////////////////////////////////////
        boost::int64_t replenish_credits(gid_type& gid)
        {
            boost::int64_t added_credit = 0;

            {
                typedef gid_type::mutex_type::scoped_lock scoped_lock;
                scoped_lock l(gid);
                HPX_ASSERT(0 == get_credit_from_gid(gid));

                added_credit = naming::detail::fill_credit_for_gid(gid);
                naming::detail::set_credit_split_mask_for_gid(gid);
            }

            gid_type unlocked_gid = gid;        // strips lock-bit

            return agas::incref(unlocked_gid, added_credit);
        }

        boost::int64_t add_credit_to_gid(gid_type& id, boost::int64_t credits)
        {
            boost::int64_t c = get_credit_from_gid(id);

            c += credits;
            set_credit_for_gid(id, c);

            return c;
        }

        boost::int64_t remove_credit_from_gid(gid_type& id, boost::int64_t debit)
        {
            boost::int64_t c = get_credit_from_gid(id);
            HPX_ASSERT(c > debit);

            c -= debit;
            set_credit_for_gid(id, c);

            return c;
        }

        boost::int64_t fill_credit_for_gid(gid_type& id,
            boost::int64_t credits)
        {
            boost::int64_t c = get_credit_from_gid(id);
            HPX_ASSERT(c <= credits);

            boost::int64_t added = credits - c;
            set_credit_for_gid(id, credits);

            return added;
        }

        ///////////////////////////////////////////////////////////////////////
        struct gid_serialization_data
        {
            gid_type gid_;
            detail::id_type_management type_;

            template <class Archive>
            void serialize(Archive& ar, unsigned)
            {
                ar & gid_ & type_;
            }
        };

        // serialization
        void id_type_impl::save(serialization::output_archive& ar, unsigned) const
        {
            if(ar.is_future_awaiting())
            {
                preprocess_gid(ar);
                return;
            }

            // Avoid performing side effects if the archive is not saving the
            // data.
            if (ar.is_saving())
            {
                id_type_management type = type_;

                gid_type new_gid;
                if (unmanaged == type_)
                {
                    new_gid = *this;
                }
                else if(managed_move_credit == type_)
                {
                    // all credits will be moved to the returned gid
                    new_gid = move_gid(const_cast<id_type_impl&>(*this));
                    type = managed;
                }
                else
                {
                    new_gid = ar.get_new_gid(*this);
                    HPX_ASSERT(new_gid != invalid_gid);
                }

                gid_serialization_data data { new_gid, type };
                ar << data;
            }
            else
            {
                gid_serialization_data data { *this, type_ };
                ar << data;
            }
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

        /// support functions for boost::intrusive_ptr
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

    ///////////////////////////////////////////////////////////////////////////
    gid_type operator+ (gid_type const& lhs, gid_type const& rhs)
    {
        boost::uint64_t lsb = lhs.id_lsb_ + rhs.id_lsb_;
        boost::uint64_t msb = lhs.id_msb_ + rhs.id_msb_;

#if defined(HPX_DEBUG)
        // make sure we're using the operator+ in proper contexts only
        boost::uint64_t lhs_internal_bits =
            detail::get_internal_bits(lhs.id_msb_);

        boost::uint64_t msb_test =
            detail::strip_internal_bits_from_gid(lhs.id_msb_) +
            detail::strip_internal_bits_and_locality_from_gid(rhs.id_msb_);

        HPX_ASSERT(msb == (msb_test | lhs_internal_bits));
#endif

        if (lsb < lhs.id_lsb_ || lsb < rhs.id_lsb_)
            ++msb;

        return gid_type(msb, lsb);
    }

    gid_type operator- (gid_type const& lhs, gid_type const& rhs)
    {
        boost::uint64_t lsb = lhs.id_lsb_ - rhs.id_lsb_;
        boost::uint64_t msb = lhs.id_msb_ - rhs.id_msb_;

// #if defined(HPX_DEBUG)
//         // make sure we're using the operator- in proper contexts only
//         boost::uint64_t lhs_internal_bits = detail::get_internal_bits(lhs.id_msb_);
//
//         boost::uint64_t msb_test =
//             detail::strip_internal_bits_and_locality_from_gid(lhs.id_msb_) -
//             detail::strip_internal_bits_and_locality_from_gid(rhs.id_msb_);
//
//         boost::uint32_t lhs_locality_id =
//             naming::get_locality_id_from_gid(lhs.id_msb_);
//         boost::uint32_t rhs_locality_id =
//             naming::get_locality_id_from_gid(rhs.id_msb_);
//         if (rhs_locality_id != naming::invalid_locality_id)
//         {
//             HPX_ASSERT(lhs_locality_id == rhs_locality_id);
//             HPX_ASSERT(msb == naming::replace_locality_id(
//                 msb_test | lhs_internal_bits, naming::invalid_locality_id));
//         }
//         else
//         {
//             HPX_ASSERT(msb == naming::replace_locality_id(
//                 msb_test | lhs_internal_bits, lhs_locality_id));
//         }
// #endif

        if (lsb > lhs.id_lsb_)
            --msb;

        return gid_type(msb, lsb);
    }

    std::ostream& operator<< (std::ostream& os, gid_type const& id)
    {
        boost::io::ios_flags_saver ifs(os);
        if (id != naming::invalid_gid)
        {
            os << std::hex
               << "{" << std::right << std::setfill('0') << std::setw(16)
                      << id.id_msb_ << ", "
                      << std::right << std::setfill('0') << std::setw(16)
                      << id.id_lsb_ << "}";
        }
        else
        {
            os << "{invalid}";
        }
        return os;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    void gid_type::save(
        T& ar
      , const unsigned int version) const
    {
        ar << id_msb_ << id_lsb_;
    }

    template <typename T>
    void gid_type::load(
        T& ar
      , const unsigned int /*version*/)
    {
        ar >> id_msb_ >> id_lsb_;

        id_msb_ &= ~is_locked_mask;     // strip lock-bit upon receive
    }

    template HPX_EXPORT void gid_type::save<serialization::output_archive>(
        serialization::output_archive&
      , const unsigned int) const;
    template HPX_EXPORT void gid_type::load<serialization::input_archive>(
        serialization::input_archive&
      , const unsigned int);

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    void id_type::save(T& ar,
        const unsigned int version) const
    {
        // We serialize the intrusive ptr and use pointer tracking here. This
        // avoids multiple credit splitting if we need multiple future await
        // passes (they all work on the same archive).
        ar << gid_;
    }

    template <typename T>
    void id_type::load(T& ar,
        const unsigned int version)
    {
        ar >> gid_;
    }

    template HPX_EXPORT void id_type::save<serialization::output_archive>(
        serialization::output_archive&, const unsigned int) const;
    template HPX_EXPORT void id_type::load<serialization::input_archive>(
        serialization::input_archive&, const unsigned int);

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
    naming::id_type get_colocation_id_sync(naming::id_type const& id, error_code& ec)
    {
        return agas::get_colocation_id_sync(id, ec);
    }

    lcos::future<naming::id_type> get_colocation_id(naming::id_type const& id)
    {
        return agas::get_colocation_id(id);
    }
}

