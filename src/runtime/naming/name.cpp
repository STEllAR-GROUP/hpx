//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/actions/continuation.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>

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
            //if (gid_was_split(*p) || !p->is_resolved())
            {
                // guard for wait_abort and other shutdown issues
                try {
                    // decrement global reference count for the given gid,
                    boost::uint32_t credits = get_credit_from_gid(*p);
                    BOOST_ASSERT(0 != credits);

                    if (get_runtime_ptr())
                    {
                        error_code ec;
                        // Fire-and-forget semantics.
                        naming::get_agas_client().decref(*p, credits, ec);
                    }
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing decrement_refcnt:"
                        << e.what();
                }
            }
//             else {
//                 // If the gid was not split at any point in time we can assume
//                 // that the referenced object is fully local.
//                 components::component_type t =
//                     static_cast<components::component_type>(p->address_.type_);
//
//                 BOOST_ASSERT(t != components::component_invalid);
//                 // Third parameter is the count of how many components to destroy.
//                 components::stubs::runtime_support::free_component_sync(t, *p, 1);
//             }
            delete p;   // delete local gid representation in any case
        }


        // custom deleter for managed gid_types, will be called when the last
        // copy of the corresponding naming::id_type goes out of scope
        void gid_managed_deleter (id_type_impl* p)
        {
            // a credit of zero means the component is not (globally) reference
            // counted
            boost::uint32_t credits = get_credit_from_gid(*p);
            if (0 != credits)
            {
                // We take over the ownership of the gid_type object here
                // as the shared_ptr is assuming it has been properly deleted
                // already. The actual deletion happens in the decrement_refcnt
                // once it is executed.
                //error_code ec;
                //applier::register_work(boost::bind(decrement_refcnt, p),
                //    "decrement global gid reference count",
                //    threads::thread_state(threads::pending),
                //    threads::thread_priority_normal, std::size_t(-1), ec);
                //if (ec)
                //{
                    // if we are not able to spawn a new thread, we need to execute
                    // the deleter directly
                    decrement_refcnt(p);
                //}
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

        void gid_transmission_deleter (id_type_impl* p)
        {
            delete p;   // delete local gid representation only
        }

//         bool id_type_impl::is_local_cached()
//         {
//             applier::applier& appl = applier::get_applier();
//             gid_type::mutex_type::scoped_lock l(this);
//             return address_ ? address_.locality_ == appl.here() : false;
//         }
//
//         bool id_type_impl::is_cached() const
//         {
//             gid_type::mutex_type::scoped_lock l(this);
//             return address_ ? true : false;
//         }
//
//         bool id_type_impl::is_local()
//         {
//             bool valid = false;
//             {
//                 gid_type::mutex_type::scoped_lock l(this);
//                 valid = address_ ? true : false;
//             }
//
//             if (!valid && !resolve())
//                 return false;
//
//             // is_local_cached()
//             applier::applier& appl = applier::get_applier();
//             gid_type::mutex_type::scoped_lock l(this);
//             return address_.locality_ == appl.here();
//         }
//
//         // check just local cache_
//         bool id_type_impl::is_local_c_cache()
//         {
//             bool valid = false;
//             {
//                 gid_type::mutex_type::scoped_lock l(this);
//                 valid = address_ ? true : false;
//             }
//
//             if (!valid && !resolve_c())
//                 return false;
//
//             applier::applier& appl = applier::get_applier();
//             gid_type::mutex_type::scoped_lock l(this);
//             return address_.locality_ == appl.here();
//         }
//
//         bool id_type_impl::resolve(naming::address& addr)
//         {
//             bool valid = false;
//             {
//                 gid_type::mutex_type::scoped_lock l(this);
//                 valid = address_ ? true : false;
//             }
//
//             // if it already has been resolved, just return the address
//             if (!valid && !resolve())
//                 return false;
//
//             addr = address_;
//             return true;
//         }
//
//         bool id_type_impl::resolve_c(naming::address& addr)
//         {
//             bool valid = false;
//             {
//                 gid_type::mutex_type::scoped_lock l(this);
//                 valid = address_ ? true : false;
//             }
//
//             // if it already has been resolved, just return the address
//             if (!valid && !resolve_c())
//                 return false;
//
//             addr = address_;
//             return true;
//         }
//
//         bool id_type_impl::resolve()
//         {
//             // call only if not already resolved
//             applier::applier& appl = applier::get_applier();
//
//             error_code ec;
//             address addr;
//             if (appl.get_agas_client().resolve(*this, addr, true, ec) && !ec)
//             {
//                 gid_type::mutex_type::scoped_lock l(this);
//                 address_ = addr;
//                 return true;
//             }
//             return false;
//         }
//
//         bool id_type_impl::resolve_c()
//         {
//             // call only if not already resolved
//             applier::applier& appl = applier::get_applier();
//
//             error_code ec;
//             address addr;
//             if (appl.get_agas_client().resolve_cached(*this, addr, ec) && !ec)
//             {
//                 gid_type::mutex_type::scoped_lock l(this);
//                 address_ = addr;
//                 return true;
//             }
//             return false;
//         }

            if (detail::unknown_deleter == m) {
            }


            type_ = m;
        }

        // explicit instantiation for the correct archive types
        template HPX_EXPORT void id_type_impl::save(
            util::portable_binary_oarchive&, const unsigned int version) const;

        template HPX_EXPORT void id_type_impl::load(
            util::portable_binary_iarchive&, const unsigned int version);
    }   // detail

    template <class Archive>
    void id_type::save(Archive& ar, const unsigned int version) const
    {
        bool isvalid = gid_ ? true : false;
        ar << isvalid;
        if (isvalid) {
            gid_type const& g = *gid_;
            detail::id_type_management m = gid_->get_management_type();
            ar << m;
            ar << g;
        }
    }

    template <class Archive>
    void id_type::load(Archive& ar, const unsigned int version)
    {
        if (version > HPX_IDTYPE_VERSION)
        {
            HPX_THROW_EXCEPTION(version_too_new, "id_type::load",
                "trying to load id_type of unknown version");
        }

        bool isvalid;
        ar >> isvalid;
        if (isvalid) {
            detail::id_type_management m;
            ar >> m;

            if (detail::unknown_deleter == m)
            {
                HPX_THROW_EXCEPTION(version_too_new, "id_type::load",
                    "trying to load id_type with unknown deleter");
            }

            if (detail::transmission == m)
                m = detail::managed;

            gid_type g;
            ar >> g;
            gid_.reset(new detail::id_type_impl(g, m));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
    template HPX_EXPORT void
    id_type::save(util::portable_binary_oarchive&, const unsigned int version) const;

    template HPX_EXPORT void
    id_type::load(util::portable_binary_iarchive&, const unsigned int version);

    ///////////////////////////////////////////////////////////////////////////
    char const* const management_type_names[] =
    {
        "unknown_deleter",    // -1
        "unmanaged",          // 0
        "managed",            // 1
        "transmission"        // 2
    };

    char const* get_management_type_name(id_type::management_type m)
    {
        if (m < id_type::unknown_deleter || m > id_type::transmission)
            return "invalid";
        return management_type_names[m + 1];
    }
}}

