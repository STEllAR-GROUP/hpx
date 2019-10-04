//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_LOCALITY_HPP
#define HPX_PARCELSET_POLICIES_MPI_LOCALITY_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)

#include <hpx/plugins/parcelport/mpi/mpi_environment.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <boost/io/ios_state.hpp>

#include <cstdint>

namespace hpx { namespace parcelset
{
    namespace policies { namespace mpi
    {
        class locality
        {
        public:
            locality()
              : rank_(-1)
            {}

            explicit locality(std::int32_t rank)
              : rank_(rank)
            {}

            std::int32_t rank() const
            {
                return rank_;
            }

            static const char *type()
            {
                return "mpi";
            }

            explicit operator bool() const noexcept
            {
                return rank_ != -1;
            }

            void save(serialization::output_archive & ar) const
            {
                ar << rank_;
            }

            void load(serialization::input_archive & ar)
            {
                ar >> rank_;
            }

        private:
            friend bool operator==(locality const & lhs, locality const & rhs)
            {
                return lhs.rank_ == rhs.rank_;
            }

            friend bool operator<(locality const & lhs, locality const & rhs)
            {
                return lhs.rank_ < rhs.rank_;
            }

            friend std::ostream & operator<<(std::ostream & os, locality const & loc)
            {
                boost::io::ios_flags_saver ifs(os);
                os << loc.rank_;

                return os;
            }

            std::int32_t rank_;
        };
    }}
}}

#endif

#endif

