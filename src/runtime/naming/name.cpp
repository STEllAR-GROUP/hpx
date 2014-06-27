//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/exception.hpp>
#include <hpx/state.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/base_object.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/agas/interface.hpp>

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/serialization/array.hpp>

#include <boost/mpl/bool.hpp>

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
// exhausted (reaches the value '2') it has to be replenished. This operation
// is performed synchronously. This is done  to ensure that AGAS has accounted
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

namespace boost { namespace serialization
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
                    else if (!threads::threadmanager_is(hpx::stopping)) {
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
        naming::gid_type id_type_impl::preprocess_gid(
            boost::uint32_t dest_locality_id) const
        {
            // unmanaged gids do not require any special handling
            if (unmanaged == type_)
                return *this;

            HPX_ASSERT(has_credits(*this));

            // Request new credits from AGAS if needed (i.e. the remainder
            // of the credit splitting is equal to one).
            if (managed == type_)
                return split_gid_if_needed(const_cast<id_type_impl&>(*this));

            // all credits will be moved to the returned gid
            HPX_ASSERT(managed_move_credit == type_);
            return move_gid(const_cast<id_type_impl&>(*this));
        }

        ///////////////////////////////////////////////////////////////////////
        gid_type split_gid_if_needed(gid_type& gid)
        {
            gid_type::mutex_type::scoped_try_lock l(gid.get_mutex());
            if (l)
            {
                // split credit normally
                return split_gid_if_needed_locked(gid);
            }

            // Just replenish the credit of the new gid and don't touch the
            // local gid instance. This is less efficient than necessary but
            // avoids deadlocks during serialization.
            return replenish_new_gid_if_needed_locked(gid);
        }

        gid_type split_gid_if_needed_locked(gid_type& gid)
        {
            naming::gid_type new_gid;

            if (naming::detail::has_credits(gid))
            {
                new_gid = naming::detail::split_credits_for_gid(gid);

                boost::int64_t src_credit =
                    naming::detail::get_credit_from_gid(gid);

                // none of the ids should be left without credits
                HPX_ASSERT(src_credit != 0);
                HPX_ASSERT(detail::has_credits(new_gid));

                // Credit exhaustion - we need to get more.
                if (1 == src_credit)
                {
                    HPX_ASSERT(1 == naming::detail::get_credit_from_gid(new_gid));

                    boost::int64_t added_credit =
                        naming::detail::fill_credit_for_gid(gid);

                    naming::gid_type unlocked_gid = gid;
                    hpx::future<boost::int64_t> f1 =
                        agas::incref_async(unlocked_gid, added_credit);

                    boost::int64_t added_new_credit =
                        naming::detail::fill_credit_for_gid(new_gid);
                    hpx::future<boost::int64_t> f2 =
                        agas::incref_async(new_gid, added_new_credit);

                    hpx::wait_all(f1, f2);
                }
            }
            else
            {
                new_gid = gid;        // strips lock-bit
            }

            return new_gid;
        }

        gid_type replenish_new_gid_if_needed_locked(gid_type const& gid)
        {
            naming::gid_type new_gid = gid;     // strips lock bit

            if (naming::detail::has_credits(new_gid))
            {
                naming::detail::strip_credits_from_gid(new_gid);
                boost::int64_t added_credit =
                    naming::detail::fill_credit_for_gid(new_gid);
                naming::detail::set_credit_split_mask_for_gid(new_gid);
                agas::incref(new_gid, added_credit);
            }

            return new_gid;
        }

        ///////////////////////////////////////////////////////////////////////
        gid_type move_gid(gid_type& gid)
        {
            gid_type::mutex_type::scoped_try_lock l(gid.get_mutex());
            if (l)
            {
                // move credit normally
                return move_gid_locked(gid);
            }

            // Just replenish the credit of the new gid and don't touch the
            // local gid instance. This is less efficient than necessary but
            // avoids deadlocks during serialization.
            return replenish_new_gid_if_needed_locked(gid);
        }

        gid_type move_gid_locked(gid_type& gid)
        {
            naming::gid_type new_gid = gid;        // strips lock-bit

            if (naming::detail::has_credits(gid))
            {
                naming::detail::strip_credits_from_gid(gid);
            }

            return new_gid;
        }

        ///////////////////////////////////////////////////////////////////////
        boost::int64_t replenish_credits(gid_type& gid)
        {
            boost::int64_t added_credit = 0;

            gid_type::mutex_type::scoped_lock l(gid);

            HPX_ASSERT(0 == get_credit_from_gid(gid));
            added_credit = naming::detail::fill_credit_for_gid(gid);
            naming::detail::set_credit_split_mask_for_gid(gid);

            gid_type unlocked_gid = gid;        // strips lock-bit
            return agas::incref(unlocked_gid, added_credit);
        }

        ///////////////////////////////////////////////////////////////////////
        struct gid_serialization_data
        {
            gid_type gid_;
            boost::uint16_t type_;
        };

        // serialization
        void id_type_impl::save(util::portable_binary_oarchive& ar) const
        {
            boost::uint32_t dest_locality_id = ar.get_dest_locality_id();

            id_type_management type = type_;
            if (managed_move_credit == type)
                type = managed;

            if(ar.flags() & util::disable_array_optimization) {
                naming::gid_type split_id(preprocess_gid(dest_locality_id));
                ar << split_id << type;
            }
            else {
                gid_serialization_data data;
                data.gid_ = preprocess_gid(dest_locality_id);
                data.type_ = type;

                ar.save(data);
            }
        }

        void id_type_impl::load(util::portable_binary_iarchive& ar)
        {
            if(ar.flags() & util::disable_array_optimization) {
                // serialize base class and management type
                ar >> static_cast<gid_type&>(*this);
                ar >> type_;
            }
            else {
                gid_serialization_data data;
                ar.load(data);

                static_cast<gid_type&>(*this) = data.gid_;
                type_ = static_cast<id_type_management>(data.type_);
            }

            if (detail::unmanaged != type_ && detail::managed != type_) {
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
    void gid_type::save(
        util::portable_binary_oarchive& ar
      , const unsigned int version) const
    {
        if(ar.flags() & util::disable_array_optimization)
            ar << id_msb_ << id_lsb_;
        else
            ar.save(*this);
    }

    void gid_type::load(
        util::portable_binary_iarchive& ar
      , const unsigned int /*version*/)
    {
        if(ar.flags() & util::disable_array_optimization)
            ar >> id_msb_ >> id_lsb_;
        else
            ar.load(*this);

        id_msb_ &= ~is_locked_mask;     // strip lock-bit upon receive
    }

    ///////////////////////////////////////////////////////////////////////////
    void id_type::save(util::portable_binary_oarchive& ar,
        const unsigned int version) const
    {
        bool isvalid = gid_ != 0;
        ar.save(isvalid);
        if (isvalid)
            gid_->save(ar);
    }

    void id_type::load(util::portable_binary_iarchive& ar,
        const unsigned int version)
    {
        if (version > HPX_IDTYPE_VERSION) {
            HPX_THROW_EXCEPTION(version_too_new, "id_type::load",
                "trying to load id_type of unknown version");
        }

        bool isvalid;
        ar.load(isvalid);
        if (isvalid) {
            boost::intrusive_ptr<detail::id_type_impl> gid(
                new detail::id_type_impl);
            gid->load(ar);
            std::swap(gid_, gid);
        }
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
    naming::id_type get_colocation_id_sync(naming::id_type const& id, error_code& ec)
    {
        return agas::get_colocation_id_sync(id, ec);
    }

    lcos::future<naming::id_type> get_colocation_id(naming::id_type const& id)
    {
        return agas::get_colocation_id(id);
    }
}

