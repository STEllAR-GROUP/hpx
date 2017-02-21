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
// array of uint32_t rather than uint8_t/char so we can easily access
// the ip for debug/validation purposes
// --------------------------------------------------------------------
struct locality {

    // the number of 32bit ints stored in our array
    static const uint32_t array_length = HPX_PARCELPORT_LIBFABRIC_LOCALITY_SIZE/4;
	static const uint32_t array_size = HPX_PARCELPORT_LIBFABRIC_LOCALITY_SIZE;

    // array type of our locality data
    typedef std::array<uint32_t, array_length> locality_data;

    static const char *type() {
        return "libfabric";
    }

    explicit locality(const locality_data &in_data)
    {
        std::memcpy(&data_[0], &in_data[0], array_size);
        LOG_DEBUG_MSG("explicit constructing locality from " << ipaddress(ip_address()) << ":" << decnumber(port()));
    }

    locality() {
        std::memset(&data_[0], 0x00, array_size);
        LOG_DEBUG_MSG("default constructing locality from " << ipaddress(ip_address()) << ":" << decnumber(port()));
    }

    locality(const locality &other) : data_(other.data_) {
        LOG_DEBUG_MSG("copy constructing locality with " << ipaddress(ip_address()) << ":" << decnumber(port()));
    }

    locality(locality &&other) : data_(std::move(other.data_)) {
        LOG_DEBUG_MSG("move constructing locality with " << ipaddress(ip_address()) << ":" << decnumber(port()));
    }

    locality & operator = (const locality &other) {
    	data_            = other.data_;
        LOG_DEBUG_MSG("copy operator locality with " << ipaddress(ip_address()) << ":" << decnumber(port()));
        return *this;
    }

    const uint32_t & ip_address() const {
        return data_[1];
    }

	uint16_t port() const {
        uint16_t port = 256*reinterpret_cast<const uint8_t*>(data_.data())[2]
			+ reinterpret_cast<const uint8_t*>(data_.data())[3];
        return port;
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

    const void *fabric_data() const { return data_.data(); }

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

