//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/locality_result.hpp>
#include <hpx/include/serialization.hpp>

#include <boost/serialization/vector.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    ///
    locality_result_iterator::data::data(result_type::const_iterator begin,
            result_type::const_iterator end)
      : current_(begin), end_(end), is_at_end_(begin == end)
    {
        if (!is_at_end_)
            current_gid_ = (*current_).begin();
    }

    /// construct end iterator
    locality_result_iterator::data::data()
      : is_at_end_(true)
    {}

    void locality_result_iterator::data::increment()
    {
        if (!is_at_end_) {
            if (++current_gid_ == (*current_).end()) {
                if (++current_ != end_) {
                    current_gid_ = (*current_).begin();
                }
                else {
                    is_at_end_ = true;
                }
            }
        }
    }

    bool locality_result_iterator::data::equal(data const& rhs) const
    {
        if (is_at_end_ != rhs.is_at_end_)
            return false;

        return (is_at_end_ && rhs.is_at_end_) ||
               (current_ == rhs.current_ && current_gid_ == rhs.current_gid_);
    }

    naming::id_type const& locality_result_iterator::data::dereference() const
    {
        BOOST_ASSERT(!is_at_end_);
        return *current_gid_;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// return an iterator range for the given locality_result's
    std::pair<locality_result_iterator, locality_result_iterator>
    locality_results(std::vector<util::locality_result> const& v)
    {
        typedef std::pair<locality_result_iterator, locality_result_iterator>
            result_type;
        return result_type(locality_result_iterator(v), locality_result_iterator());
    }
}}

HPX_REGISTER_BASE_LCO_WITH_VALUE(
    std::vector<hpx::util::remote_locality_result>,
    factory_locality_result)
