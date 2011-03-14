//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>

#include <hpx/hpx_fwd.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/fusion/include/at.hpp>

#include <hpx/util/logging.hpp>
#include <hpx/runtime/naming/server/reply.hpp>
#include <hpx/runtime/naming/server/request.hpp>
#include <hpx/runtime/naming/server/request_handler.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/lcos/mutex.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    template <typename Mutex>
    inline void init_mutex(Mutex& m)
    {
    }
    inline void init_mutex(boost::detail::spinlock& m)
    {
        boost::detail::spinlock l = BOOST_DETAIL_SPINLOCK_INIT;
        m = l;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    request_handler<Mutex>::request_handler()
      : totals_(command_lastcommand), 
        console_prefix_(0),
        component_type_(components::component_first_dynamic)
    {
        init_mutex(ns_registry_mtx_);
        init_mutex(registry_mtx_);
        init_mutex(component_types_mtx_);
    }

    template <typename Mutex>
    request_handler<Mutex>::~request_handler()
    {}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_getprefix(request const& req, reply& rep, 
        bool self)
    {
        try {
            typename mutex_type::scoped_lock l(registry_mtx_);
            site_prefix_map_type::iterator it = 
                site_prefixes_.find(req.get_site());
            if (it != site_prefixes_.end()) {
                // The real prefix has to be used as the 32 most 
                // significant bits of global id's

                // verify that this locality is either the only console of the 
                // given application or a worker 
                if (self && req.isconsole()) {
                    if (0 != console_prefix_ && console_prefix_ != (*it).second.first) {
                        rep = reply(self ? command_getprefix : command_getprefix_for_site, 
                            duplicate_console);
                        return;
                    }
                    console_prefix_ = (*it).second.first;
                }

                // existing entry
                rep = reply(repeated_request, 
                    self ? command_getprefix : command_getprefix_for_site,
                    get_gid_from_prefix((*it).second.first)); 
            }
            else {
                // insert this prefix as being mapped to the given locality
                boost::uint32_t prefix = (boost::uint32_t)(site_prefixes_.size() + 1);
                naming::gid_type id = get_gid_from_prefix(prefix);

                // verify that this locality is either the only console of the 
                // given application or a worker 
                if (self && req.isconsole()) {
                    if (0 != console_prefix_) {
                        rep = reply(self ? command_getprefix : command_getprefix_for_site, 
                            duplicate_console);
                        return;
                    }
                    console_prefix_ = prefix;
                }

                // start assigning ids with the second block of 64Bit numbers only
                naming::gid_type lower_id (id.get_msb() + 1, 0);
                site_prefixes_.insert(
                    site_prefix_map_type::value_type(req.get_site(), 
                        std::make_pair(prefix, lower_id)));

                // now, bind this prefix to the locality address allowing to 
                // send parcels to the memory of a locality
                registry_type::iterator it = registry_.find(id);
                if (it != registry_.end()) {
                    // this shouldn't happen
                    rep = reply(self ? command_getprefix : command_getprefix_for_site, 
                        no_success, "prefix is already bound to local address");
                    return;
                }
                else {
                    // the zero as last parameter means 'all' lsb gids
                    registry_.insert(registry_type::value_type(id, 
                        registry_data_type(address(
                            req.get_site(), components::component_runtime_support), 
                        1, 0)));
                }

                // The real prefix has to be used as the 32 most 
                // significant bits of global id's

                // created new entry
                rep = reply(success, 
                    self ? command_getprefix : command_getprefix_for_site, id);
            }
        }
        catch (std::bad_alloc) {
            rep = reply(self ? command_getprefix : command_getprefix_for_site, 
                out_of_memory);
        }
        catch (...) {
            rep = reply(self ? command_getprefix : command_getprefix_for_site, 
                internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_getconsoleprefix(request const& req, reply& rep)
    {
        try {
            typename mutex_type::scoped_lock l(registry_mtx_);

            if (0 != console_prefix_) {
                rep = reply(success, command_getconsoleprefix, 
                    get_gid_from_prefix(console_prefix_)); 
            }
            else {
                rep = reply(command_getconsoleprefix, no_registered_console,
                    "no console prefix registered for this application"); 
            }
        }
        catch (std::bad_alloc) {
            rep = reply(command_getconsoleprefix, out_of_memory);
        }
        catch (...) {
            rep = reply(command_getconsoleprefix, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Mutex>
        inline void 
        lock_helper(Mutex&, typename Mutex::scoped_lock& l)
        {
            l.lock();
        }
        inline void 
        lock_helper(boost::detail::spinlock& m, boost::detail::spinlock::scoped_lock&)
        {
            m.lock();
        }

        template <typename Mutex>
        inline void 
        unlock_helper(Mutex&, typename Mutex::scoped_lock& l)
        {
            l.unlock();
        }
        inline void 
        unlock_helper(boost::detail::spinlock& m, boost::detail::spinlock::scoped_lock&)
        {
            m.unlock();
        }

        template <typename Mutex>
        struct unlocker
        {
            unlocker(Mutex& m, typename Mutex::scoped_lock& l) 
              : m_(m), l_(l) 
            {
                unlock_helper(m, l);
            }
            ~unlocker()
            {
                lock_helper(m_, l_);
            }

            Mutex& m_;
            typename Mutex::scoped_lock& l_;
        };
    }

    template <typename Mutex>
    void request_handler<Mutex>::handle_getprefixes(request const& req, reply& rep)
    {
        try {
            typename mutex_type::scoped_lock l(registry_mtx_);

            std::vector<boost::uint32_t> prefixes;
            prefixes.reserve(site_prefixes_.size());

            components::component_type t = req.get_type();
            if (components::component_invalid == t) {
                // return all prefixes
                typename site_prefix_map_type::iterator end = site_prefixes_.end();
                for (typename site_prefix_map_type::iterator it = site_prefixes_.begin();
                     it != end; ++it)
                {
                    prefixes.push_back((*it).second.first);
                }
            }
            else {
                detail::unlocker<Mutex> ul(registry_mtx_, l);

                // return prefixes which have a factory for the given type
                typename mutex_type::scoped_lock fl(component_types_mtx_);
                std::pair<factory_map::iterator, factory_map::iterator> p = 
                    factories_.equal_range(t);
                for (/**/; p.first != p.second; ++p.first) 
                {
                    prefixes.push_back((*p.first).second);
                }
            }

            rep = reply(prefixes, prefixes.empty() ? no_success : success); 
        }
        catch (std::bad_alloc) {
            rep = reply(command_getprefixes, out_of_memory);
        }
        catch (...) {
            rep = reply(command_getprefixes, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_get_component_id(request const& req, reply& rep)
    {
        try {
            typename mutex_type::scoped_lock l(component_types_mtx_);

            component_type_map::iterator it = component_types_.find(req.get_name());
            if (it == component_types_.end()) {
                // first request: create a new component type and store it
                std::pair<typename component_type_map::iterator, bool> p = 
                    component_types_.insert(component_type_map::value_type(
                        req.get_name(), component_type_++));
                if (!p.second) {
                    rep = reply(command_get_component_id, out_of_memory);
                    return;
                }
                it = p.first;
            }

            // return the registered component type
            rep = reply(command_get_component_id, 
                (components::component_type)(*it).second); 
        }
        catch (std::bad_alloc) {
            rep = reply(command_get_component_id, out_of_memory);
        }
        catch (...) {
            rep = reply(command_get_component_id, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_register_factory(request const& req, reply& rep)
    {
        try {
            typename mutex_type::scoped_lock l(component_types_mtx_);

            // ensure component type
            component_type_map::iterator it = component_types_.find(req.get_name());
            if (it == component_types_.end()) {
                // first request: create a new component type and store it
                std::pair<typename component_type_map::iterator, bool> p = 
                    component_types_.insert(component_type_map::value_type(
                        req.get_name(), component_type_++));
                if (!p.second) {
                    rep = reply(command_register_factory, out_of_memory);
                    return;
                }
                it = p.first;
            }

            // store prefix in factory map
            boost::uint32_t prefix = get_prefix_from_gid(req.get_id());
            typename factory_map::iterator itf = factories_.insert(
                typename factory_map::value_type((*it).second, prefix));

            if (itf == factories_.end()) 
            {
                rep = reply(command_register_factory, out_of_memory);
                return;
            }

            // return the registered component type
            rep = reply(command_register_factory, 
                (components::component_type)(*it).second); 
        }
        catch (std::bad_alloc) {
            rep = reply(command_register_factory, out_of_memory);
        }
        catch (...) {
            rep = reply(command_register_factory, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_getidrange(request const& req, reply& rep)
    {
        try {
            typename mutex_type::scoped_lock l(registry_mtx_);
            typename site_prefix_map_type::iterator it = 
                site_prefixes_.find(req.get_site());
            if (it != site_prefixes_.end()) {
                // The real prefix has to be used as the 32 most 
                // significant bits of global id's

                LAGAS_(debug) << "handle_getidrange: reusing site: " 
                    << req.get_site() << ", prefix: " << (*it).second.first
                    << ", current upper: " << (*it).second.second;

                // generate the new id range
                naming::gid_type lower ((*it).second.second + 1);
                naming::gid_type upper (lower + (req.get_count() - 1));

                if (upper.get_msb() != lower.get_msb()) {
                    // handle overflow
                    if ((lower.get_msb() & ~0xFFFFFFFF) == 0xFFFFFFFF)
                    {
                        rep = reply(command_getidrange, internal_server_error, 
                            "global ids have been exhausted");
                        return;
                    }
                    lower = naming::gid_type(upper.get_msb(), 0);
                    upper = lower + (req.get_count() - 1);
                }

                // store the new lower bound
                (*it).second.second = upper;

                LAGAS_(debug) << "handle_getidrange: new upper: " 
                    << (*it).second.second;

                // set initial credits
                naming::set_credit_for_gid(lower, HPX_INITIAL_GLOBALCREDIT);
                naming::set_credit_for_gid(upper, HPX_INITIAL_GLOBALCREDIT);

                // existing entry
                rep = reply(repeated_request, command_getidrange, lower, upper); 
            }
            else {
                // insert this prefix as being mapped to the given locality
                boost::uint32_t prefix = (boost::uint32_t)site_prefixes_.size() + 1;
                naming::gid_type id = get_gid_from_prefix(prefix);

                // start assigning ids with the second block of 64Bit numbers only
                naming::gid_type lower_id (id.get_msb() + 1, 0);

                LAGAS_(debug) << "handle_getidrange: new site: " 
                    << req.get_site() << ", prefix: " << id
                    << ", current upper: " << lower_id;

                std::pair<typename site_prefix_map_type::iterator, bool> p =
                    site_prefixes_.insert(
                        typename site_prefix_map_type::value_type(req.get_site(), 
                            std::make_pair(prefix, lower_id)));

                // make sure the entry got created
                if (!p.second) {
                    if (LAGAS_ENABLED(debug)) {
                        BOOST_FOREACH(typename site_prefix_map_type::value_type v, site_prefixes_)
                        {
                            LAGAS_(debug) 
                                << "handle_getidrange: registered site: "
                                << v.first;
                        }
                    }
                    rep = reply(command_getidrange, no_success, 
                        "couldn't create site prefix map entry");
                    return;
                }

                // now, bind this prefix to the locality address allowing to 
                // send parcels to the memory of a locality
                typename registry_type::iterator it = registry_.find(id);
                if (it != registry_.end()) {
                    // this shouldn't happen
                    rep = reply(command_getidrange, no_success, 
                        "prefix is already bound to local address");
                    return;
                }
                else {
                    registry_.insert(typename registry_type::value_type(id, 
                        registry_data_type(address(
                            req.get_site(), components::component_runtime_support), 
                        1, 0)));
                }

                // generate the new id range
                naming::gid_type lower = lower_id + 1;
                naming::gid_type upper = lower + (req.get_count() - 1);

                // store the new lower bound
                (*p.first).second.second = upper;

                LAGAS_(debug) << "handle_getidrange: new upper: " 
                    << (*p.first).second.second;

                // set initial credits
                naming::set_credit_for_gid(lower, HPX_INITIAL_GLOBALCREDIT);
                naming::set_credit_for_gid(upper, HPX_INITIAL_GLOBALCREDIT);

                // created new entry
                rep = reply(success, command_getidrange, lower, upper);
            }
        }
        catch (std::bad_alloc) {
            rep = reply(command_getidrange, out_of_memory);
        }
        catch (...) {
            rep = reply(command_getidrange, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_bind_range(request const& req, reply& rep)
    {
        try {
            error s = no_success;
            std::string str;
            {
                using boost::fusion::at_c;

                typename mutex_type::scoped_lock l(registry_mtx_);
                naming::gid_type id = req.get_id();
                naming::strip_credit_from_gid(id);

                typename registry_type::iterator it = registry_.lower_bound(id);

                if (it != registry_.end()) {
                    if ((*it).first == id) {
                        // update existing bindings
                        if (at_c<1>((*it).second) != req.get_count()) 
                        {
                            // this is an error since we can't change block sizes 
                            s = bad_parameter;
                            str = "can't change block size or number of gids per "
                                  "object of existing binding";
                        }
                        else {
                            // store the new address and offsets
                            at_c<0>((*it).second) = req.get_address();
                            at_c<2>((*it).second) = req.get_offset();
                        }
                    }
                    else if (it != registry_.begin()) {
                        --it;
                        if ((*it).first + at_c<1>((*it).second) > id) {
                            // the previous range covers the new id
                            s = bad_parameter;
                            str = "the new global id is contained in an existing range";
                        }
                        else {
                            // create new bindings
                            create_new_binding(req, id, s, str);
                        }
                    }
                    else {
                        // create new bindings, the existing ranges are larger 
                        // than the new global id
                        create_new_binding(req, id, s, str);
                    }
                }
                else {
                    if (!registry_.empty()) {
                        --it;
                        if ((*it).first + at_c<1>((*it).second) > id) {
                            // the previous range covers the new id
                            s = bad_parameter;
                            str = "the new global id is contained in an existing range";
                        }
                        else {
                            // create new bindings
                            create_new_binding(req, id, s, str);
                        }
                    }
                    else {
                        // create new bindings
                        create_new_binding(req, id, s, str);
                    }
                }
            }
            rep = reply(command_bind_range, s, str.empty() ? NULL : str.c_str());
        }
        catch (std::bad_alloc) {
            rep = reply(command_bind_range, out_of_memory);
        }
        catch (...) {
            rep = reply(command_bind_range, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_incref(request const& req, reply& rep)
    {
        try {
            typename mutex_type::scoped_lock l(registry_mtx_);

            naming::gid_type id = req.get_id();
            naming::strip_credit_from_gid(id);

            typename refcnt_store_type::iterator it = refcnts_.find(id);
            if (it == refcnts_.end()) 
            {
                // we insert a new reference count entry with an initial 
                // count of HPX_INITIAL_GLOBALCREDIT because we assume bind() 
                // has already created the first reference (plus credits).
                std::pair<typename refcnt_store_type::iterator, bool> p = 
                    refcnts_.insert(typename refcnt_store_type::value_type(id, 
                                        HPX_INITIAL_GLOBALCREDIT));
                if (!p.second)
                {
                    rep = reply(command_incref, out_of_memory);
                    return;
                }
                it = p.first;
            }

            rep = reply(command_incref, (*it).second += req.get_count());
        }
        catch (std::bad_alloc) {
            rep = reply(command_incref, out_of_memory);
        }
        catch (...) {
            rep = reply(command_incref, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    components::component_type
    request_handler<Mutex>::get_component_type(naming::gid_type const& gid)
    {
        using boost::fusion::at_c;

        typename mutex_type::scoped_lock l(registry_mtx_);
        typename registry_type::iterator it = registry_.lower_bound(gid);
        if (it != registry_.end()) {
            if ((*it).first == gid) {
                // found the exact match in the registry
                components::component_type t = (components::component_type)at_c<0>((*it).second).type_;
                return (components::component_type)at_c<0>((*it).second).type_;
            }
            else if (it != registry_.begin()) {
                --it;
                if ((*it).first + at_c<1>((*it).second) > gid) {
                    // the previous range covers the given global id

                    // the only limitation while binding blocks of global 
                    // ids is that these have to have identical msb's
                    if (gid.get_msb() == (*it).first.get_msb()) {
                        components::component_type t = (components::component_type)at_c<0>((*it).second).type_;
                        return (components::component_type)at_c<0>((*it).second).type_;
                    }
                }
            }
        }
        else if (!registry_.empty()) {
            --it;
            if ((*it).first + at_c<1>((*it).second) >= gid) {
                // the previous range covers the id to resolve
                components::component_type t = (components::component_type)at_c<0>((*it).second).type_;
                return (components::component_type)at_c<0>((*it).second).type_;
            }
        }
        return components::component_invalid;
    }

    template <typename Mutex>
    void request_handler<Mutex>::handle_decref(request const& req, reply& rep)
    {
        try {
            naming::gid_type id = req.get_id();
            naming::strip_credit_from_gid(id);

            BOOST_ASSERT(req.get_count() <= HPX_INITIAL_GLOBALCREDIT);

            boost::uint64_t cnt = 0;

            {
                typename mutex_type::scoped_lock l(registry_mtx_);
                typename refcnt_store_type::iterator it = refcnts_.find(id);
                if (it != refcnts_.end()) 
                {
                    if ((*it).second < req.get_count()) {
                        rep = reply(command_decref, bad_parameter,
                            "Bogus credit while decrementing global reference count.");
                        return;
                    }

                    (*it).second -= req.get_count();
                    cnt = (*it).second;

                    if (0 == cnt)
                        refcnts_.erase(it);   // last reference removes entry
                }
                else if (req.get_count() < HPX_INITIAL_GLOBALCREDIT) 
                {
                    // we insert a new reference count entry with an initial 
                    // count of HPX_INITIAL_GLOBALCREDIT because we assume bind() 
                    // has already created the first reference (plus credits).
                    std::pair<typename refcnt_store_type::iterator, bool> p = 
                        refcnts_.insert(
                            typename refcnt_store_type::value_type(id, HPX_INITIAL_GLOBALCREDIT));
                    if (!p.second)
                    {
                        rep = reply(command_decref, out_of_memory);
                        return;
                    }
                    it = p.first;

                    if ((*it).second < req.get_count()) {
                        rep = reply(command_decref, bad_parameter,
                            "Bogus credit while decrementing global reference count.");
                        return;
                    }

                    (*it).second -= req.get_count();
                    cnt = (*it).second;

                    // FIXME: It should be impossible for this if statement
                    // to be true, because this entire branch is only entered if
                    // the requested count is less than HPX_INITIAL_GLOBALCREDIT.
                    // Shouldn't we just assert here?
                    //if (0 == cnt)
                    //    refcnts_.erase(it);   // last reference removes entry
                }
            }

            components::component_type t = components::component_invalid;
            if (0 == cnt) 
            {
                t = get_component_type(id);
                if (t == components::component_invalid) {
                    rep = reply(command_decref, bad_component_type,
                        "Unknown component type while decrementing global reference count.");
                    return;
                }
            }
            rep = reply(command_decref, cnt, t);
        }
        catch (std::bad_alloc) {
            rep = reply(command_decref, out_of_memory);
        }
        catch (...) {
            rep = reply(command_decref, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_unbind_range(request const& req, reply& rep)
    {
        try {
            error s = no_success;
            std::string str;
            {
                using boost::fusion::at_c;

                typename mutex_type::scoped_lock l(registry_mtx_);

                naming::gid_type id = req.get_id();
                naming::strip_credit_from_gid(id);

                typename registry_type::iterator it = registry_.find(id);
                if (it != registry_.end()) {
                    if (at_c<1>((*it).second) != req.get_count()) {
                        // this is an error since we can't use a different 
                        // block size while unbinding
                        s = bad_parameter;
                        str = "block sizes must match";
                    }
                    else {
                        rep = reply(command_unbind_range, at_c<0>((*it).second));
                        registry_.erase(it);
                        return;
                    }
                }
            }
            rep = reply(command_unbind_range, s, 
                str.empty() ? NULL : str.c_str());
        }
        catch (std::bad_alloc) {
            rep = reply(command_unbind_range, out_of_memory);
        }
        catch (...) {
            rep = reply(command_unbind_range, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_resolve(request const& req, reply& rep)
    {
        try {
            using boost::fusion::at_c;

            typename mutex_type::scoped_lock l(registry_mtx_);

            naming::gid_type id = req.get_id();
            naming::strip_credit_from_gid(id);

            typename registry_type::iterator it = registry_.lower_bound(id);
            if (it != registry_.end()) {
                if ((*it).first == id) {
                    // found the exact match in the registry
                    rep = reply(command_resolve, at_c<0>((*it).second));
                }
                else if (it != registry_.begin()) {
                    --it;
                    if ((*it).first + at_c<1>((*it).second) > id) {
                        // the previous range covers the given global id

                        // the only limitation while binding blocks of global 
                        // ids is that these have to have identical msb's
                        if (id.get_msb() != (*it).first.get_msb()) {
                            // no existing range covers the given global id
                            rep = reply(command_resolve, internal_server_error,
                                "msb's of global ids should match");
                        }
                        else {
                            // calculate the local address corresponding to the 
                            // given global id
                            naming::address addr (at_c<0>((*it).second));
                            boost::uint64_t gid_offset = 
                                id.get_lsb() - (*it).first.get_lsb();
                            addr.address_ += gid_offset * at_c<2>((*it).second);
                            rep = reply(command_resolve, addr);
                        }
                    }
                    else {
                        // no existing range covers the given global id
                        rep = reply(command_resolve, no_success);
                    }
                }
                else {
                    // all existing entries are larger than the given global id
                    rep = reply(command_resolve, no_success);
                }
            }
            else if (!registry_.empty()) {
                --it;
                if ((*it).first + at_c<1>((*it).second) >= id) {
                    // the previous range covers the id to resolve
                    naming::address addr (at_c<0>((*it).second));
                    boost::uint64_t gid_offset = 
                        id.get_lsb() - (*it).first.get_lsb();
                    addr.address_ += gid_offset * at_c<2>((*it).second);
                    rep = reply(command_resolve, addr);
                }
                else {
                    rep = reply(command_resolve, no_success);
                }
            }
            else {
                rep = reply(command_resolve, no_success);
            }
        }
        catch (std::bad_alloc) {
            rep = reply(command_resolve, out_of_memory);
        }
        catch (...) {
            rep = reply(command_resolve, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_queryid(request const& req, reply& rep)
    {
        try {
            typename mutex_type::scoped_lock l(ns_registry_mtx_);
            typename ns_registry_type::iterator it = ns_registry_.find(req.get_name());
            if (it != ns_registry_.end()) 
                rep = reply(command_queryid, (*it).second);
            else
                rep = reply(command_queryid, no_success);
        }
        catch (std::bad_alloc) {
            rep = reply(command_queryid, out_of_memory);
        }
        catch (...) {
            rep = reply(command_queryid, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_registerid(request const& req, reply& rep)
    {
        try {
            error s = no_success;
            {
                typename mutex_type::scoped_lock l(ns_registry_mtx_);

                naming::gid_type id = req.get_id();
                naming::strip_credit_from_gid(id);

                typename ns_registry_type::iterator it = ns_registry_.find(req.get_name());
                if (it != ns_registry_.end())
                    (*it).second = id;
                else {
                    ns_registry_.insert(typename ns_registry_type::value_type(
                        req.get_name(), id));
                    s = success;    // created new entry
                }
            }
            rep = reply(command_registerid, s);
        }
        catch (std::bad_alloc) {
            rep = reply(command_registerid, out_of_memory);
        }
        catch (...) {
            rep = reply(command_registerid, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_unregisterid(request const& req, reply& rep)
    {
        try {
            error s = no_success;
            {
                typename mutex_type::scoped_lock l(ns_registry_mtx_);
                typename ns_registry_type::iterator it = ns_registry_.find(req.get_name());
                if (it != ns_registry_.end()) {
                    ns_registry_.erase(it);
                    s = success;
                }
            }
            rep = reply(command_unregisterid, s);
        }
        catch (std::bad_alloc) {
            rep = reply(command_unregisterid, out_of_memory);
        }
        catch (...) {
            rep = reply(command_unregisterid, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
#if BOOST_VERSION < 103600
    double extract_count(std::pair<double, std::size_t> const& p)
    {
        return double(p.second);
    }
#endif
    
    template <typename Mutex>
    void request_handler<Mutex>::handle_statistics_count(request const& req, reply& rep)
    {
        try {
#if BOOST_VERSION >= 103600
            rep = reply(command_statistics_count, totals_, boost::accumulators::count);
#else
            rep = reply(command_statistics_count, totals_, extract_count);
#endif
        }
        catch (std::bad_alloc) {
            rep = reply(command_statistics_count, out_of_memory);
        }
        catch (...) {
            rep = reply(command_statistics_count, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
#if BOOST_VERSION < 103600
    double extract_mean(std::pair<double, std::size_t> const& p)
    {
        return (0 != p.second) ? p.first / p.second : 0.0;
    }
#endif
    
    template <typename Mutex>
    void request_handler<Mutex>::handle_statistics_mean(request const& req, reply& rep)
    {
        try {
#if BOOST_VERSION >= 103600
            rep = reply(command_statistics_mean, totals_, boost::accumulators::mean);
#else
            rep = reply(command_statistics_mean, totals_, extract_mean);
#endif
        }
        catch (std::bad_alloc) {
            rep = reply(command_statistics_mean, out_of_memory);
        }
        catch (...) {
            rep = reply(command_statistics_mean, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
#if BOOST_VERSION >= 103600
    template <typename Mutex>
    double request_handler<Mutex>::extract_moment2(accumulator_set_type const& p)
    {
        return boost::accumulators::extract::moment<2>(p);
    }
#else
    double extract_moment2(std::pair<double, std::size_t> const& p)
    {
        return 0.0;   // not implemented yet
    }
#endif

    template <typename Mutex>
    void request_handler<Mutex>::handle_statistics_moment2(request const& req, reply& rep)
    {
        try {
#if BOOST_VERSION >= 103600
            rep = reply(command_statistics_moment2, totals_, 
                &request_handler<Mutex>::extract_moment2);
#else
            rep = reply(command_statistics_moment2, totals_, extract_moment2);
#endif
        }
        catch (std::bad_alloc) {
            rep = reply(command_statistics_moment2, out_of_memory);
        }
        catch (...) {
            rep = reply(command_statistics_moment2, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    void request_handler<Mutex>::handle_requests(std::vector<request> const& reqs, 
        std::vector<reply>& reps)
    {
        BOOST_FOREACH(request const& req, reqs)
        {
            reply rep;
            handle_request(req, rep);
            reps.push_back(rep);
        }
    }

    template <typename Mutex>
    void request_handler<Mutex>::handle_request(request const& req, reply& rep)
    {
        LAGAS_(info) << "request: " << req;

        switch (req.get_command()) {
        case command_getprefix:
            handle_getprefix(req, rep, true);
            break;

        case command_getprefix_for_site:
            handle_getprefix(req, rep, false);
            break;

        case command_getconsoleprefix:
            handle_getconsoleprefix(req, rep);
            break;

        case command_getprefixes:
            handle_getprefixes(req, rep);
            break;

        case command_get_component_id:
            handle_get_component_id(req, rep);
            break;

        case command_register_factory:
            handle_register_factory(req, rep);
            break;

        case command_getidrange:
            handle_getidrange(req, rep);
            break;

        case command_bind_range:
            handle_bind_range(req, rep);
            break;

        case command_incref:
            handle_incref(req, rep);
            break;

        case command_decref:
            handle_decref(req, rep);
            break;

        case command_unbind_range:
            handle_unbind_range(req, rep);
            break;

        case command_resolve:
            handle_resolve(req, rep);
            break;

        case command_queryid:
            handle_queryid(req, rep);
            break;

        case command_registerid:
            handle_registerid(req, rep);
            break;

        case command_unregisterid:
            handle_unregisterid(req, rep);
            break;

        case command_statistics_count:
            handle_statistics_count(req, rep);
            break;

        case command_statistics_mean:
            handle_statistics_mean(req, rep);
            break;

        case command_statistics_moment2:
            handle_statistics_moment2(req, rep);
            break;

        default:
            rep = reply(bad_request);
            break;
        }

        if (rep.get_status() != success && rep.get_status() != repeated_request) {
            LAGAS_(error) << "response: " << rep;
        }
        else {
            LAGAS_(info) << "response: " << rep;
        }
    }

}}}  // namespace hpx::naming::server

///////////////////////////////////////////////////////////////////////////////
template class hpx::naming::server::request_handler<boost::mutex>;
// template hpx::naming::server::request_handler<hpx::lcos::mutex>;
template class hpx::naming::server::request_handler<boost::detail::spinlock>;
