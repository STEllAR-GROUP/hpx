////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_RUNTIME_THREADS_POLICIES_PARSE_AFFINITY_OPTIONS_HPP
#define HPX_RUNTIME_THREADS_POLICIES_PARSE_AFFINITY_OPTIONS_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_HWLOC)
#include <hpx/exception_fwd.hpp>
#include <hpx/util/assert.hpp>

#include <boost/cstdint.hpp>
#include <boost/variant.hpp>

#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace threads { namespace detail
{
    typedef std::vector<boost::int64_t> bounds_type;

    enum distribution_type
    {
        compact  = 0x01,
        scatter  = 0x02,
        balanced = 0x04
    };

    struct spec_type
    {
        enum type { unknown, thread, socket, numanode, core, pu };
        HPX_API_EXPORT static char const* type_name(type t);

        static boost::int64_t all_entities()
        {
            return (std::numeric_limits<boost::int64_t>::min)();
        }

        spec_type(type t = unknown, boost::int64_t min = all_entities(),
                boost::int64_t max = all_entities())
          : type_(t), index_bounds_()
        {
            if (t != unknown) {
                if (max == 0 || max == all_entities()) {
                    // one or all entities
                    index_bounds_.push_back(min);
                }
                else if (min != all_entities()) {
                    // all entities between min and -max, or just min,max
                    HPX_ASSERT(min >= 0);
                    index_bounds_.push_back(min);
                    index_bounds_.push_back(max);
                }
            }
        }

        bool operator==(spec_type const& rhs) const
        {
            if (type_ != rhs.type_ || index_bounds_.size() !=
                rhs.index_bounds_.size())
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
    typedef boost::variant<distribution_type, mappings_spec_type> mappings_type;

    HPX_API_EXPORT bounds_type extract_bounds(spec_type& m,
        std::size_t default_last, error_code& ec);

    HPX_API_EXPORT void parse_mappings(std::string const& spec,
        mappings_type& mappings, error_code& ec = throws);
}}}
#endif

#endif // HPX_RUNTIME_THREADS_POLICIES_PARSE_AFFINITY_OPTIONS_HPP
