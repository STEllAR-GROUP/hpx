//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_LOCALITY_HPP
#define HPX_PARCELSET_POLICIES_VERBS_LOCALITY_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/util/safe_bool.hpp>

namespace hpx { namespace parcelset {
    namespace policies { namespace verbs
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

      explicit locality(boost::uint32_t ip) :
            ip_(ip), qp_(0xFFFF) {}

      locality() : ip_(0xFFFF), qp_(0xFFFF) {}

      // some condition marking this locality as valid
      operator util::safe_bool<locality>::result_type() const {
        return util::safe_bool<locality>()(ip_ != boost::uint32_t(0xFFFF));
      }

      void save(serialization::output_archive & ar) const {
        ar.save(ip_);
        ar.save(qp_);
      }

      void load(serialization::input_archive & ar) {
        ar.load(ip_);
        ar.load(qp_);
      }

    private:
      friend bool operator==(locality const & lhs, locality const & rhs) {
        return (lhs.ip_ == rhs.ip_) && (lhs.qp_ == rhs.qp_);
      }

      friend bool operator<(locality const & lhs, locality const & rhs) {
        return lhs.ip_ < rhs.ip_;
      }

      friend std::ostream & operator<<(std::ostream & os, locality const & loc) {
        boost::io::ios_flags_saver
        ifs(os);
        os << loc.ip_ << " : " << os << loc.qp_;
        return os;
      }
    public:
      boost::uint32_t ip_;
      boost::uint32_t qp_;
    };

}}}}

#endif

