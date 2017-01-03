//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_LOCALITY_HPP
#define HPX_PARCELSET_POLICIES_VERBS_LOCALITY_HPP

#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
//
#include <cstdint>

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{

    // --------------------------------------------------------------------
    // Locality, represented by an ip address and a queue pair (qp) id
    // the qp id is used internally for quick lookup to find the peer
    // that we want to communicate with
    // --------------------------------------------------------------------
    struct locality {
      static const char *type() {
        return "verbs";
      }

      explicit locality(std::uint32_t ip) :
            ip_(ip)
      {
          LOG_DEBUG_MSG("explicit constructing locality from " << hexuint32(ip))
      }

      locality() : ip_(0xFFFF)
      {
          LOG_DEBUG_MSG("constructing locality from " << hexuint32(0xFFFF))
      }

      // some condition marking this locality as valid
      explicit operator bool() const {
        return (ip_ != std::uint32_t(0xFFFF));
      }

      void save(serialization::output_archive & ar) const {
        ar << ip_;
      }

      void load(serialization::input_archive & ar) {
        ar >> ip_;
      }

    private:
      friend bool operator==(locality const & lhs, locality const & rhs) {
        return (lhs.ip_ == rhs.ip_);
      }

      friend bool operator<(locality const & lhs, locality const & rhs) {
        return lhs.ip_ < rhs.ip_;
      }

      friend std::ostream & operator<<(std::ostream & os, locality const & loc) {
        boost::io::ios_flags_saver ifs(os);
        os << loc.ip_;
        return os;
      }
    public:
      std::uint32_t ip_;
    };

}}}}

#endif

