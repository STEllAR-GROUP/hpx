//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <boost/accumulators/framework/accumulator_base.hpp>
#include <boost/accumulators/framework/depends_on.hpp>
#include <boost/accumulators/framework/extractor.hpp>
#include <boost/accumulators/framework/parameters/sample.hpp>
#include <boost/accumulators/numeric/functional.hpp>
#include <boost/accumulators/statistics/rolling_window.hpp>
#include <boost/accumulators/statistics_fwd.hpp>

namespace hpx { namespace util { namespace detail {
    template <typename Sample>
    struct rolling_max_impl : boost::accumulators::accumulator_base
    {
        typedef Sample float_type;

        // for boost::result_of
        typedef float_type result_type;

        template <typename Args>
        rolling_max_impl(Args const& /* args */)
          : max_(boost::numeric::as_min(Sample()))
          , is_dirty_(false)
        {
        }

        template <typename Args>
        void operator()(Args const& args)
        {
            boost::accumulators::find_accumulator<
                boost::accumulators::tag::rolling_window>(
                args[boost::accumulators::accumulator])(args);
            is_dirty_ = true;
        }

        template <typename Args>
        result_type result(Args const& args) const
        {
            if (is_dirty_)
            {
                using namespace boost::accumulators;

                max_ = boost::numeric::as_min(Sample());

                // work around problem in older Boost versions
                auto r = rolling_window_plus1(args);
                bool full = impl::is_rolling_window_plus1_full(args);

                for (auto const& s : r.advance_begin(full ? 1 : 0))
                {
                    boost::numeric::max_assign(max_, s);
                }

                is_dirty_ = false;
            }
            return max_;
        }

    private:
        mutable Sample max_;
        mutable bool is_dirty_;
    };
}}}    // namespace hpx::util::detail

///////////////////////////////////////////////////////////////////////////////
// tag::rolling_max
namespace boost { namespace accumulators {
    namespace tag {
        struct rolling_max : depends_on<rolling_window>
        {
            struct impl
            {
                template <typename Sample, typename Weight>
                struct apply
                {
                    typedef hpx::util::detail::rolling_max_impl<Sample> type;
                };
            };
        };
    }    // namespace tag

    ///////////////////////////////////////////////////////////////////////////////
    // extract::rolling_max
    namespace extract {
        extractor<tag::rolling_max> const rolling_max = {};
    }
}}    // namespace boost::accumulators

namespace hpx { namespace util {
    namespace tag {
        using boost::accumulators::tag::rolling_max;
    }

    using boost::accumulators::extract::rolling_max;
}}    // namespace hpx::util
