//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_ID_TYPE_OCT_13_2013_0751PM)
#define HPX_NAMING_ID_TYPE_OCT_13_2013_0751PM

#include <hpx/config.hpp>
#include <hpx/config/warnings_prefix.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/safe_bool.hpp>

#include <boost/intrusive_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    namespace detail
    {
        struct HPX_EXPORT id_type_impl;
        HPX_EXPORT void intrusive_ptr_add_ref(id_type_impl* p);
        HPX_EXPORT void intrusive_ptr_release(id_type_impl* p);
    }

    struct HPX_EXPORT gid_type;

    ///////////////////////////////////////////////////////////////////////////
    // the local gid is actually just a wrapper around the real thing
    struct HPX_EXPORT id_type
    {
    private:
        friend struct detail::id_type_impl;

        id_type(detail::id_type_impl* p)
          : gid_(p)
        {}

    public:
        enum management_type
        {
            unknown_deleter = -1,
            unmanaged = 0,          ///< unmanaged GID
            managed = 1,            ///< managed GID
            managed_move_credit = 2 ///< managed GID which will give up all
                                    ///< credits when sent
        };

        id_type() {}

        id_type(boost::uint64_t lsb_id, management_type t);
        id_type(gid_type const& gid, management_type t);
        id_type(boost::uint64_t msb_id, boost::uint64_t lsb_id, management_type t);

        id_type(id_type const & o) : gid_(o.gid_) {}
        id_type(id_type && o)
          : gid_(std::move(o.gid_))
        {
            o.gid_.reset();
        }

        id_type & operator=(id_type const & o)
        {
            if (this != &o)
                gid_ = o.gid_;
            return *this;
        }
        id_type & operator=(id_type && o)
        {
            if (this != &o)
            {
                gid_ = o.gid_;
                o.gid_.reset();
            }
            return *this;
        }

        gid_type& get_gid();
        gid_type const& get_gid() const;

        // This function is used in AGAS unit tests and application code, do not
        // remove.
        management_type get_management_type() const;

        id_type& operator++();
        id_type operator++(int);

        operator util::safe_bool<id_type>::result_type() const;

        // comparison is required as well
        friend bool operator== (id_type const& lhs, id_type const& rhs);
        friend bool operator!= (id_type const& lhs, id_type const& rhs);

        friend bool operator< (id_type const& lhs, id_type const& rhs);
        friend bool operator<= (id_type const& lhs, id_type const& rhs);
        friend bool operator> (id_type const& lhs, id_type const& rhs);
        friend bool operator>= (id_type const& lhs, id_type const& rhs);

        // access the internal parts of the gid
        boost::uint64_t get_msb() const;
        void set_msb(boost::uint64_t msb);

        boost::uint64_t get_lsb() const;
        void set_lsb(boost::uint64_t lsb);
        void set_lsb(void* lsb);

        // Convert this id into an unmanaged one (in-place) - Use with maximum
        // care, or better, don't use this at all.
        void make_unmanaged() const;

    private:
        friend HPX_API_EXPORT gid_type get_parcel_dest_gid(id_type const& id);

        friend std::ostream& operator<< (std::ostream& os, id_type const& id);

        friend class hpx::serialization::access;

        template <class T>
        void save(T& ar, const unsigned int version) const;

        template <class T>
        void load(T& ar, const unsigned int version);

        HPX_SERIALIZATION_SPLIT_MEMBER()

        boost::intrusive_ptr<detail::id_type_impl> gid_;
    };

    ///////////////////////////////////////////////////////////////////////////
    id_type const invalid_id = id_type();
}}

#endif

