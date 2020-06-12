//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)

#include <hpx/modules/mpi_base.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/util/ios_flags_saver.hpp>

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
                hpx::util::ios_flags_saver ifs(os);
                os << loc.rank_;

                return os;
            }

            std::int32_t rank_;
        };
    }}
}}

#endif


