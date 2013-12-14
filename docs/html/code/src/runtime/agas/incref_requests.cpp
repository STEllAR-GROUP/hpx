//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/incref_requests.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/lcos/future.hpp>

#include <boost/foreach.hpp>

namespace hpx { namespace agas { namespace detail
{
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

        mutex_type::scoped_lock l(mtx_);

        // Either add the given credit to the local entry or create
        // a new entry.
        iterator it = find_entry(gid);
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
            }
            else if (data.keep_alive_.get_management_type() ==
                     naming::id_type::unmanaged)
            {
                // If the currently stored id is unmanaged then the entry
                // was created by a remote acknowledgment. Update the
                // id with a local one to be kept alive.
                data.keep_alive_ = id;
            }
        }
        else
        {
            store_.insert(incref_requests_type::value_type(
                gid, incref_request_data(credits, id)));
        }
    }

    // This function will be called during id-splitting to store a bread-crumb
    // pointing to the locality where the outstanding credit has to be tracked.
    bool incref_requests::add_remote_incref_request(boost::int64_t credits,
        naming::gid_type const& gid, boost::uint32_t remote_locality)
    {
        HPX_ASSERT(credits > 0);

        mutex_type::scoped_lock l(mtx_);

        // There is nothing for us to do if no local entry exists.
        iterator it_local = find_entry(gid);
        if (it_local == store_.end())
            return false;

        // Subtract the given credit from any existing local entry.
        incref_request_data& data_local = it_local->second;
        if (data_local.credit_ < 0)
        {
            // this entry holds some pre-acknowledged credits only
            return false;
        }

        // Remove the maximally possible amount of credits from the local
        // entry. If there are more credits moved off the locality than there
        // are credits to be acknowledged (which can happen because of the
        // initial credit every id gets assigned when being created), we can
        // still assume that there all credits which are still waiting to be
        // acknowledged have been moved off this locality, i.e. no local
        // credits require any acknowledgment anymore.
        if (data_local.credit_ >= credits)
        {
            data_local.credit_ -= credits;
        }
        else
        {
            data_local.credit_ = 0;
        }

        // Remove the current entry when it does not hold any pending requests
        // anymore.
        if (data_local.credit_ == 0 && data_local.debit_ == 0)
        {
            store_.erase(it_local);
        }

        // Add the given credit to an existing (remote) entry or create a new
        // (remote) one.
        iterator it_remote = find_entry(gid, remote_locality);
        if (it_remote != store_.end())
        {
            HPX_ASSERT(it_remote->second.credit_ > 0);
            it_remote->second.credit_ += credits;
        }
        else
        {
            store_.insert(incref_requests_type::value_type(
                gid, incref_request_data(credits, remote_locality)));
        }

        return true;
    }

    bool incref_requests::acknowledge_request(boost::int64_t credits,
        naming::id_type const& id, acknowledge_request_callback const& f)
    {
        HPX_ASSERT(credits > 0);
        naming::gid_type gid = naming::detail::get_stripped_gid(id.get_gid());

        std::vector<incref_request_data> matching_data;
        std::vector<incref_request_data> decref_data;

        {
            mutex_type::scoped_lock l(mtx_);

            std::pair<iterator, iterator> r = store_.equal_range(gid);
            while (r.first != r.second && credits != 0)
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

                    // any pending decref requests will be handled once all
                    // incref requests are acknowledged
                    matching_data.back().debit_ = 0;

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
                        // This entry also stores some pending decref request.
                        // We're allowed to release those decref requests only
                        // if there is no other entry referring this id besides
                        // the current one anymore.
                        std::pair<iterator, iterator> r = store_.equal_range(gid);
                        if (r.first != r.second)
                        {
                            // Reinsert the data as local pure decref requests
                            decref_data.push_back(matching_data.back());
                            decref_data.back().credit_ = 0;

                            // Don't release the decref request right now.
                            matching_data.back().debit_ = 0;
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

        // now handle all acknowledged credits and all delayed debits
        std::vector<hpx::future<bool> > requests;
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
                    f(-data.debit_, raw, naming::invalid_locality_id)
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
        BOOST_FOREACH(hpx::future<bool>& f, requests)
        {
            result = f.get() && result;
        }

        return result;
    }

    bool incref_requests::add_decref_request(boost::int64_t credits,
        naming::gid_type const& gid)
    {
        mutex_type::scoped_lock l(mtx_);

        // We have to hold back any decref requests as long as there exists
        // some (local or remote) entry referencing the given id.
        iterator it_local = find_entry(gid);
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
                store_.insert(incref_requests_type::value_type(
                    gid, incref_request_data(credits, id)));

                return true; // hold back the decref
            }

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
            return false;
        }

        it_local->second.debit_ += credits;
        return true;
    }
}}}

