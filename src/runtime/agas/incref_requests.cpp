//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/incref_requests.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/lcos/future.hpp>

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/io/ios_state.hpp>

#include <iosfwd>

// make it easy to highlight logging from this file
#define LINCREF_        LDEB_
#define LINCREF_ENABLED LDEB_ENABLED
// #define LINCREF_        LAGAS_(info)
// #define LINCREF_ENABLED LAGAS_ENABLED(info)

namespace hpx { namespace agas { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    std::ostream& operator<<(std::ostream& os, incref_request_data const& data)
    {
        boost::io::ios_flags_saver ifs(os);
        naming::gid_type gid;
        if (data.keep_alive_ != naming::invalid_id)
            gid = naming::detail::get_stripped_gid(data.keep_alive_.get_gid());

        os << "{"
                << "credit: 0x" 
                << std::hex << std::setw(16) << std::setfill('0') << data.credit_
                << ", id: " << gid
                << ", locality: ";

        if (data.locality_ != naming::invalid_locality_id)
            os << data.locality_;
        else
            os << "{invalid}";

        os << ", debit: " << data.debit_ << "}";
        return os;
    }

    ///////////////////////////////////////////////////////////////////////////
    incref_requests::iterator incref_requests::find_entry(
        naming::gid_type const& gid, boost::uint32_t loc)
    {
        std::pair<iterator, iterator> r = store_.equal_range(gid);

        for (/**/; r.first != r.second; ++r.first)
        {
            if (r.first->second.locality_ == loc)
            {
                return r.first;
            }
        }

        return store_.end();
    }

    void incref_requests::add_incref_request(boost::int64_t credits,
        naming::id_type const& id)
    {
        HPX_ASSERT(credits > 0);

        naming::gid_type gid = naming::detail::get_stripped_gid(id.get_gid());

        LINCREF_ << (boost::format(
            "incref_requests::add_incref_request: gid(%1%): credit(%2$#016x)") %
                gid % credits);

        mutex_type::scoped_lock l(mtx_);

        // Either add the given credit to the local entry or create
        // a new entry.
        iterator it = find_entry(gid);      // locate local entry
        if (it != store_.end())
        {
            incref_request_data& data = it->second;
            data.credit_ += credits;

            if (data.credit_ == 0)
            {
                // An entry with a negative credit should not have been used to
                // store a pending decref request.
                HPX_ASSERT(data.debit_ == 0);

                // This credit request was already acknowledged earlier,
                // remove the entry.
                store_.erase(it);

                l.unlock();

                LINCREF_ << (boost::format(
                    "> incref_requests::add_incref_request: gid(%1%): "
                    "deleted existing entry") % gid);
            }
            else if (data.keep_alive_.get_management_type() ==
                     naming::id_type::unmanaged)
            {
                // If the currently stored id is unmanaged then the entry
                // was created by a remote acknowledgment. Update the
                // id with a local one to be kept alive.
                data.keep_alive_ = id;

                l.unlock();

                LINCREF_ << (boost::format(
                    "> incref_requests::add_incref_request: gid(%1%): "
                    "updated keep_alive id(%2%)") % gid % data);
            }
            else
            {
                l.unlock();

                LINCREF_ << (boost::format(
                    "> incref_requests::add_incref_request: gid(%1%): "
                    "updated existing entry(%2%)") % gid % data);
            }
        }
        else
        {
            it = store_.insert(incref_requests_type::value_type(
                gid, incref_request_data(credits, id)));

            l.unlock();
            LINCREF_ << (boost::format(
                "> incref_requests::add_incref_request: gid(%1%): "
                "created new entry(%2%)") % gid % it->second);
        }
    }

    // This function will be called during id-splitting to store a bread-crumb
    // pointing to the locality where the outstanding credit has to be tracked.
    bool incref_requests::add_remote_incref_request(boost::int64_t credits,
        naming::gid_type const& gid, boost::uint32_t remote_locality)
    {
        HPX_ASSERT(credits > 0);

        naming::gid_type raw = naming::detail::get_stripped_gid(gid);

        LINCREF_ << (boost::format(
            "incref_requests::add_remote_incref_request: gid(%1%): "
            "credit(%2$#016x), remote_locality(%3%)") %
                raw % credits % remote_locality);

        mutex_type::scoped_lock l(mtx_);

        // There is nothing for us to do if no local entry exists.
        iterator it_local = find_entry(raw);    // locate local entry
        if (it_local == store_.end())
        {
            l.unlock();

            LINCREF_ << (boost::format(
                "> incref_requests::add_remote_incref_request: gid(%1%): "
                "passing through as no local entry exists") % raw);

            return false;
        }

        // Subtract the given credit from any existing local entry.
        incref_request_data& data_local = it_local->second;
        if (data_local.credit_ < 0)
        {
            // An entry with a negative credit should not have been used to
            // store a pending decref request.
            HPX_ASSERT(data_local.debit_ == 0);

            // this entry holds some pre-acknowledged credits only
            l.unlock();

            LINCREF_ << (boost::format(
                "> incref_requests::add_remote_incref_request: gid(%1%): "
                "passing through because of pre-acknowledged credits(%2%)") %
                    raw % data_local);

            return false;
        }

        // Remove the maximally possible amount of credits from the local
        // entry. If there are more credits moved off the locality than there
        // are credits to be acknowledged (which can happen because of the
        // initial credit every id gets assigned when being created), we can
        // still assume that no local credits require acknowledgment anymore.
        if (data_local.credit_ >= credits)
        {
            data_local.credit_ -= credits;
        }
        else
        {
            data_local.credit_ = 0;
        }

        // Remove the current (local) entry when it does not hold any pending
        // requests anymore.
        bool erased_local_entry = false;
        if (data_local.credit_ == 0 && data_local.debit_ == 0)
        {
            store_.erase(it_local);
            erased_local_entry = true;
        }

        // Add the given credit to an existing (remote) entry or create a new
        // (remote) one.
        bool created_remote_entry = false;
        iterator it_remote = find_entry(raw, remote_locality);
        if (it_remote != store_.end())
        {
            HPX_ASSERT(it_remote->second.credit_ > 0);
            it_remote->second.credit_ += credits;
        }
        else
        {
            it_remote = store_.insert(incref_requests_type::value_type(
                raw, incref_request_data(credits, remote_locality)));
            created_remote_entry = true;
        }

        if (LINCREF_ENABLED)
        {
            l.unlock();

            if (erased_local_entry)
            {
                LINCREF_ << (boost::format(
                    "> incref_requests::add_remote_incref_request: gid(%1%): "
                    "erased local entry") % raw);
            }
            if (created_remote_entry)
            {
                LINCREF_ << (boost::format(
                    "> incref_requests::add_remote_incref_request: gid(%1%): "
                    "created remote entry(%2%)") % raw % it_remote->second);
            }
            else
            {
                LINCREF_ << (boost::format(
                    "> incref_requests::add_remote_incref_request: gid(%1%): "
                    "updated remote entry(%2%)") % raw % it_remote->second);
            }
        }
        return true;
    }

    bool incref_requests::acknowledge_request(boost::int64_t credits,
        naming::id_type const& id, acknowledge_request_callback const& f)
    {
        HPX_ASSERT(credits > 0);
        naming::gid_type gid = naming::detail::get_stripped_gid(id.get_gid());

        LINCREF_ << (boost::format(
            "incref_requests::acknowledge_request: gid(%1%): "
            "credit(%2$#016x)") % gid % credits);

        std::vector<incref_request_data> matching_data;
        std::vector<incref_request_data> decref_data;

        boost::int64_t debits = credits;

        {
            mutex_type::scoped_lock l(mtx_);

            std::pair<iterator, iterator> r = store_.equal_range(gid);
            while (r.first != r.second && (credits != 0 || debits != 0))
            {
                incref_request_data& data = r.first->second;
                if (data.credit_ > credits)
                {
                    // This entry is requesting more credits than have been
                    // acknowledged.

                    // adjust remaining part of the credit
                    data.credit_ -= credits;

                    // construct proper acknowledgment data
                    matching_data.push_back(data);

                    // not all pending credits were acknowledged
                    matching_data.back().credit_ = credits;

                    // For local entries, handle as many decref requests as we
                    // have received.
                    if (debits != 0 && data.locality_ == naming::invalid_locality_id)
                    {
                        if (matching_data.back().debit_ > debits)
                        {
                            // release available amount of decrefs
                            data.debit_ -= debits;
                            matching_data.back().debit_ = debits;
                            debits = 0;
                        }
                        else if (matching_data.back().debit_ != 0)
                        {
                            // all pending decref requests can be released
                            data.debit_ = 0;
                            debits -= matching_data.back().debit_;
                        }
                    }

                    // we're done with handling the acknowledged credit
                    credits = 0;

                    ++r.first;
                }
                else if (data.credit_ > 0)
                {
                    // This entry is requesting less or an equal amount of
                    // credits compared to the acknowledged amount of credits.

                    // construct proper acknowledgment data
                    matching_data.push_back(data);

                    // adjust acknowledged credits
                    credits -= data.credit_;

                    // delete the entry as it has been handled completely
                    iterator it = r.first++;
                    store_.erase(it);

                    if (matching_data.back().debit_ != 0)
                    {
                        // This entry also stores some pending decref requests.
                        // We're allowed to release those decref requests only
                        // if there is no other entry referring to this id
                        // besides the current one anymore.
                        //
                        // If there isn't any other entry we're allowed to
                        // release only as many decrefs as we have received
                        // incref acknowledgments.
                        std::pair<iterator, iterator> rr = store_.equal_range(gid);
                        if (rr.first != rr.second)
                        {
                            // Reinsert the data as local pure decref requests
                            decref_data.push_back(matching_data.back());
                            decref_data.back().credit_ = 0;

                            // Don't release the decref request right now.
                            matching_data.back().debit_ = 0;
                        }
                        else if (matching_data.back().debit_ > debits)
                        {
                            // Reinsert the data as local pure decref requests
                            // for the remaining amount.
                            decref_data.push_back(matching_data.back());
                            decref_data.back().credit_ = 0;
                            decref_data.back().debit_ =
                                matching_data.back().debit_ - debits;

                            // Release the given amount of decrefs now.
                            matching_data.back().debit_ = debits;

                            // Adjust remaining amount of decref requests.
                            debits = 0;
                        }
                        else
                        {
                            // Adjust remaining amount of decref requests as
                            // we're releasing all of the decrefs of the
                            // current entry.
                            debits -= matching_data.back().debit_;
                        }
                    }
                }
                else
                {
                    // This entry already stores a certain amount of
                    // pre-acknowledged credits.

                    // entries with zero credits should never happen
                    HPX_ASSERT(data.credit_ < 0);

                    data.credit_ -= credits;

                    // we're done with handling the acknowledged credits
                    credits = 0;

                    ++r.first;
                }
            }
            HPX_ASSERT(credits >= 0);

            if (credits != 0)
            {
                // Add negative credit entry, we expect for a matching request
                // to come in shortly.
                store_.insert(incref_requests_type::value_type(
                    gid, incref_request_data(-credits, id)));
            }

            // Re-insert deleted decref requests which need to stay.
            BOOST_FOREACH(incref_request_data const& data, decref_data)
            {
                HPX_ASSERT(data.credit_ == 0);
                HPX_ASSERT(data.debit_ != 0);

                store_.insert(incref_requests_type::value_type(gid, data));
            }
        }

        // log all pending operations
        if (LINCREF_ENABLED)
        {
            BOOST_FOREACH(incref_request_data const& data, matching_data)
            {
                LINCREF_ << (boost::format(
                    "> incref_requests::acknowledge_request: gid(%1%): "
                    "pending action(%2%)") % gid % data);
            }
        }

        // now handle all acknowledged credits and all delayed debits
        std::vector<hpx::unique_future<bool> > requests;
        
        requests.reserve(2*matching_data.size());

        BOOST_FOREACH(incref_request_data const& data, matching_data)
        {
            naming::gid_type raw = gid;
            if (data.keep_alive_)
                raw = naming::detail::get_stripped_gid(data.keep_alive_.get_gid());

            if (data.debit_ != 0)
            {
                HPX_ASSERT(data.locality_ == naming::invalid_locality_id);
                HPX_ASSERT(data.debit_ > 0);

                requests.push_back(
                    std::move(f(-data.debit_, raw, naming::invalid_locality_id))
                );
            }

            // local incref requests have already been handled above
            if (data.credit_ != 0 && data.locality_ != naming::invalid_locality_id)
            {
                requests.push_back(
                    f(data.credit_, raw, data.locality_)
                );
            }
        }

        bool result = true;
        BOOST_FOREACH(hpx::unique_future<bool>& f, requests)
        {
            result = f.get() && result;
        }

        return result;
    }

    bool incref_requests::add_decref_request(boost::int64_t credits,
        naming::gid_type const& gid)
    {
        LINCREF_ << (boost::format(
            "incref_requests::add_decref_request: gid(%1%): "
            "debit(%2$#016x)") % gid % credits);

        mutex_type::scoped_lock l(mtx_);

        // We have to hold back any decref requests as long as there exists
        // some (local or remote) entry referencing the given id.
        iterator it_local = find_entry(gid);    // locate local entry
        if (it_local == store_.end())
        {
            // no local entry exists, look for a remote one
            std::pair<iterator, iterator> r = store_.equal_range(gid);

            // perform 'normal' (non-delayed) decref handling if any entry
            // exists
            if (r.first != r.second)
            {
                // create a new (local) entry keeping track of the decref
                // request which was held back
                id_type id(gid, id_type::unmanaged);
                iterator it = store_.insert(incref_requests_type::value_type(
                    gid, incref_request_data(id, credits)));

                l.unlock();

                LINCREF_ << (boost::format(
                    "> incref_requests::add_decref_request: gid(%1%): "
                    "created new entry(%2%)") % gid % it->second);
                LINCREF_ << (boost::format(
                    "> incref_requests::add_decref_request: gid(%1%): "
                    "holding back debit(%2$#016x) because of pending remote "
                    "incref request(%3%)") % gid % credits % r.first->second);

                return true; // hold back the decref
            }

            l.unlock();

            LINCREF_ << (boost::format(
                "> incref_requests::add_decref_request: gid(%1%): "
                "passing through debit(%2$#016x)") % gid % credits);

            // no entry exists which references the given id, release the decref
            return false;
        }

        // The only way for a credit to become negative is when an
        // acknowledgment comes in and no matching incref entry is found
        // (assuming the incref request is about to come in).
        //
        // We don't need to hold back any decref request for such an id.
        if (it_local->second.credit_ < 0)
        {
            // this entry holds some pre-acknowledged credits only
            l.unlock();

            LINCREF_ << (boost::format(
                "> incref_requests::add_decref_request: gid(%1%): "
                "passing through debit(%2$#016x) because of pre-acknowledged "
                "credits(%3%)") % gid % credits % it_local->second);

            return false;
        }

        // keep track of decref requests
        it_local->second.debit_ += credits;

        l.unlock();

        LINCREF_ << (boost::format(
            "> incref_requests::add_decref_request: gid(%1%): "
            "holding back debit(%2$#016x) because of pending local incref "
            "request(%3%)") % gid % credits % it_local->second);

        return true;
    }
}}}

