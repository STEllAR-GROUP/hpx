//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_DETAIL_RESOLVER_DO_UNDO_JAN_21_2009_1245PM)
#define HPX_NAMING_DETAIL_RESOLVER_DO_UNDO_JAN_21_2009_1245PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/bulk_resolver_client.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>

#include <boost/noncopyable.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

namespace hpx { namespace naming { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    struct do_undo_base
    {
        do_undo_base() {}

        virtual ~do_undo_base() {}

        virtual void do_it(bulk_resolver_client& resolver, int index, 
            error_code& ec = throws) {}
        virtual void undo_it() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    struct do_undo_incref : public do_undo_base
    {
        typedef boost::function<void()> do_type;
        typedef boost::function<void()> undo_type;

        do_undo_incref(do_type do_it, undo_type undo_it) 
          : do_it_(do_it), undo_it_(undo_it) {}

    private:
        void do_it(bulk_resolver_client& resolver, int index, 
            error_code& ec = throws);
        void undo_it();

        do_type do_it_;
        undo_type undo_it_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct do_undo_resolve : public do_undo_base
    {
        typedef boost::function<void(naming::address)> do_type;

        do_undo_resolve(do_type do_it)
          : do_it_(do_it) {}

    private:
        void do_it(bulk_resolver_client& resolver, int index, 
            error_code& ec = throws);

        do_type do_it_;
    };

    ///////////////////////////////////////////////////////////////////////////
    class bulk_resolver_helper : boost::noncopyable
    {
    public:
        bulk_resolver_helper(resolver_client& resolver)
          : resolver_(resolver)
        {}

        int resolve(gid_type const& id, parcelset::parcel& p);
        int incref(id_type const& id, boost::uint32_t credits, 
            id_type const& oldid, boost::uint32_t oldcredits);

        void execute(error_code& ec = throws);
        void undo();

    protected:
        static void add_credit_to_gid(naming::id_type const& id, 
            boost::uint16_t credit);
        static void set_credit_for_gid(naming::id_type const& id, 
            boost::uint16_t credit);

    private:
        bulk_resolver_client resolver_;
        boost::ptr_vector<do_undo_base> tasks_;
    };
}}}

#endif



