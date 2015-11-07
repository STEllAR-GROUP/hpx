//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_PARCEL_SUFFIX_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_PARCEL_SUFFIX_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)

#include <hpx/runtime/naming/name.hpp>

#include <boost/cstdint.hpp>
#include <boost/io/ios_state.hpp>

#include "hash.hpp"

namespace hpx { namespace components { namespace security
{
#if defined(_MSC_VER)
#  pragma pack(push, 1)
#endif

    class parcel_suffix
    {
    public:
        parcel_suffix()
        {
        }

        parcel_suffix(boost::uint32_t locality_id,
                naming::gid_type const& parcel_id, hash const& hash)
          : locality_id_(locality_id), parcel_id_(parcel_id), hash_(hash)
        {
        }

        boost::uint32_t get_locality_id() const
        {
            return locality_id_;
        }

        naming::gid_type const & get_parcel_id() const
        {
            return parcel_id_;
        }

        hash const & get_hash() const
        {
            return hash_;
        }

        friend std::ostream & operator<<(std::ostream & os,
                                         parcel_suffix const & parcel_suffix)
        {
            return os << "<parcel_suffix "
                      << parcel_suffix.locality_id_
                      << " "
                      << parcel_suffix.parcel_id_
                      << " "
                      << parcel_suffix.hash_
                      << ">";
        }

        unsigned char const* begin() const
        {
            return reinterpret_cast<unsigned char const*>(this);
        }

        unsigned char const* end() const
        {
            return reinterpret_cast<unsigned char const*>(this) + size();
        }

        BOOST_CONSTEXPR static std::size_t size()
        {
            return sizeof(parcel_suffix);
        }

    private:
        boost::uint32_t locality_id_;
        naming::gid_type parcel_id_;
        hash hash_;
    };

#if defined(_MSC_VER)
#  pragma pack(pop)
#endif
}}}

#endif

#endif
