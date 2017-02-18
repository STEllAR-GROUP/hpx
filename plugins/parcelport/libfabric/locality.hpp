//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_LOCALITY_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_LOCALITY_HPP

#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
//
#include <cstdint>
#include <array>

#define HPX_PARCELPORT_LIBFABRIC_LOCALITY_SIZE 16

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{

// --------------------------------------------------------------------
// Locality, in this structure we store the informartion required by
// libfabric to make a connection to another node.
// With libfabric 1.4.x the array contains the fabric ip address stored
// as the second uint32_t in the array. For this reason we use an
// array of uint32_T rather than uint8_t/char so we can easily access
// the ip for debug/validation purposes
// --------------------------------------------------------------------
struct locality {

    // the number of 32bit ints stored in our array
    static const uint32_t array_length = HPX_PARCELPORT_LIBFABRIC_LOCALITY_SIZE/4;

    // array type of our locality data
    typedef std::array<uint32_t, array_length> locality_data;

    static const char *type() {
        return "libfabric";
    }

    explicit locality(const locality_data &in_data)
    {
        std::memcpy(&data_[0], &in_data[0], HPX_PARCELPORT_LIBFABRIC_LOCALITY_SIZE);
        uint32_t &ipaddr = data_[1];
        LOG_DEBUG_MSG("explicit constructing locality from " << ipaddress(ipaddr));
    }

    locality() {
        std::memset(&data_[0], 0x00, HPX_PARCELPORT_LIBFABRIC_LOCALITY_SIZE);
        uint32_t &ipaddr = data_[1];
        LOG_DEBUG_MSG("default constructing locality from " << ipaddress(ipaddr));
    }

    uint32_t ip_address() const {
        return data_[1];
    }

    // some condition marking this locality as valid
    explicit operator bool() const {
        const uint32_t &ipaddr = data_[1];
        return (ipaddr != 0);
    }

    void save(serialization::output_archive & ar) const {
        ar << data_;
    }

    void load(serialization::input_archive & ar) {
        ar >> data_;
    }

private:
    friend bool operator==(locality const & lhs, locality const & rhs) {
        uint32_t a1 = lhs.ip_address();
        uint32_t a2 = rhs.ip_address();
        LOG_DEBUG_MSG("Testing array equality "
            << ipaddress(a1)
            << ipaddress(a2)
        );

        return (lhs.data_ == rhs.data_);
    }

    friend bool operator<(locality const & lhs, locality const & rhs) {
        const uint32_t &ipaddr_1 = lhs.data_[1];
        const uint32_t &ipaddr_2 = rhs.data_[1];
        return ipaddr_1 < ipaddr_2;
    }

    friend std::ostream & operator<<(std::ostream & os, locality const & loc) {
        boost::io::ios_flags_saver ifs(os);
        for (uint32_t i=0; i<array_length; ++i) {
            os << loc.data_[i];
        }
        return os;
    }

private:
    locality_data data_;
};

}}}}

#endif

