//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_LOCALITY_HPP
#define HPX_PARCELSET_POLICIES_MPI_LOCALITY_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)

#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/plugins/parcelport/mpi/mpi_environment.hpp>

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

            explicit locality(boost::int32_t rank)
              : rank_(rank)
            {}

            boost::int32_t rank() const
            {
                return rank_;
            }

            static const char *type()
            {
                return "mpi";
            }

            explicit operator bool() const HPX_NOEXCEPT
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

            boost::int32_t rank_;
        };
    }}
}}

#endif

#endif

