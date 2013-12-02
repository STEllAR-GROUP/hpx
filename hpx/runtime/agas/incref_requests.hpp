//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_AGAS_INCREF_REQUESTS_DEC_01_203_0133PM)
#define HPX_RUNTIME_AGAS_INCREF_REQUESTS_DEC_01_203_0133PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <map>

namespace hpx { namespace agas { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    struct incref_request_data
    {
        incref_request_data(boost::uint64_t credit,
                naming::id_type const& id, naming::id_type const& loc)
            : credit_(credit), keep_alive_(id), locality_(loc)
        {}

        boost::int64_t credit_;         // amount of credit not acknowledged yet
        naming::id_type keep_alive_;    // id for which credit is outstanding
        naming::id_type locality_;      // locality where this credit belongs
    };

    ///////////////////////////////////////////////////////////////////////////
    class incref_requests
    {
    private:
        typedef lcos::local::spinlock mutex_type;

        typedef std::multimap<naming::gid_type, incref_request_data>
            incref_requests_type;

        typedef incref_requests_type::iterator iterator;
        typedef incref_requests_type::const_iterator const_iterator;

        iterator find_entry(naming::gid_type const& gid,
            naming::id_type const& loc);

    public:
        incref_requests() {}

        // Add a credit request for a local id. This will add the credit to any
        // existing local reuest.
        //
        // This function will be called as part of setting up an incref request.
        // It will also keep the given id alive until all of the credits are
        // acknowledged (see acknowledge_request below).
        void add_request(boost::int64_t credit, naming::id_type const& id);

        // Add a credit request for a remote id, this will subtract the credit from an
        // existing local request before adding the rmeote one. This function does
        // nothing if there is no local request for the given gid.
        //
        // This function will be called during id-splitting to store a bread-crumb
        // pointing to the locality where the outstanding credit has to be tracked.
        bool add_remote_request(boost::int64_t credit, naming::gid_type const& gid,
            naming::id_type const& remote_locality);

        typedef HPX_STD_FUNCTION<
            bool(boost::int64_t, naming::id_type const&, naming::id_type const&)
        > acknowledge_request_callback;

        bool acknowledge_request(boost::int64_t credit, naming::id_type const& id,
             acknowledge_request_callback const& f);

    private:
        mutable mutex_type mtx_;
        incref_requests_type store_;
    };
}}}

#endif
