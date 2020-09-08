////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/topology/cpu_mask.hpp>

#include <boost/variant.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace threads {
    namespace detail {
        typedef std::vector<std::int64_t> bounds_type;

        enum distribution_type
        {
            compact = 0x01,
            scatter = 0x02,
            balanced = 0x04,
            numa_balanced = 0x08
        };

        struct spec_type
        {
            enum type
            {
                unknown,
                thread,
                socket,
                numanode,
                core,
                pu
            };
            HPX_CORE_EXPORT static char const* type_name(type t);

            static std::int64_t all_entities()
            {
                return (std::numeric_limits<std::int64_t>::min)();
            }

            spec_type(type t = unknown, std::int64_t min = all_entities(),
                std::int64_t max = all_entities())
              : type_(t)
              , index_bounds_()
            {
                if (t != unknown)
                {
                    if (max == 0 || max == all_entities())
                    {
                        // one or all entities
                        index_bounds_.push_back(min);
                    }
                    else if (min != all_entities())
                    {
                        // all entities between min and -max, or just min,max
                        HPX_ASSERT(min >= 0);
                        index_bounds_.push_back(min);
                        index_bounds_.push_back(max);
                    }
                }
            }

            bool operator==(spec_type const& rhs) const
            {
                if (type_ != rhs.type_ ||
                    index_bounds_.size() != rhs.index_bounds_.size())
                    return false;

                for (std::size_t i = 0; i < index_bounds_.size(); ++i)
                {
                    if (index_bounds_[i] != rhs.index_bounds_[i])
                        return false;
                }

                return true;
            }

            type type_;
            bounds_type index_bounds_;
        };

        typedef std::vector<spec_type> mapping_type;
        typedef std::pair<spec_type, mapping_type> full_mapping_type;
        typedef std::vector<full_mapping_type> mappings_spec_type;
        typedef boost::variant<distribution_type, mappings_spec_type>
            mappings_type;

        HPX_CORE_EXPORT bounds_type extract_bounds(
            spec_type const& m, std::size_t default_last, error_code& ec);

        HPX_CORE_EXPORT void parse_mappings(std::string const& spec,
            mappings_type& mappings, error_code& ec = throws);
    }    // namespace detail

    HPX_CORE_EXPORT void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::size_t num_threads,
        std::vector<std::size_t>& num_pus, bool use_process_mask,
        error_code& ec = throws);

    // backwards compatibility helper
    inline void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, error_code& ec = throws)
    {
        std::vector<std::size_t> num_pus;
        parse_affinity_options(
            spec, affinities, 1, 1, affinities.size(), num_pus, false, ec);
    }
}}    // namespace hpx::threads
