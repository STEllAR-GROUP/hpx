//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <hpx/config.hpp>

#include <boost/range/iterator_range.hpp>
#include <boost/parameter/keyword.hpp>
#include <boost/accumulators/framework/accumulator_base.hpp>
#include <boost/accumulators/framework/extractor.hpp>
#include <boost/accumulators/numeric/functional.hpp>
#include <boost/accumulators/framework/parameters/sample.hpp>
#include <boost/accumulators/framework/depends_on.hpp>
#include <boost/accumulators/statistics_fwd.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
namespace boost { namespace accumulators
{
    ///////////////////////////////////////////////////////////////////////////
    // cache_size and num_bins named parameters
    BOOST_PARAMETER_NESTED_KEYWORD(tag, histogram_num_bins, num_bins)
    BOOST_PARAMETER_NESTED_KEYWORD(tag, histogram_min_range, min_range)
    BOOST_PARAMETER_NESTED_KEYWORD(tag, histogram_max_range, max_range)
}}

namespace hpx { namespace util
{
    using boost::accumulators::histogram_num_bins;
    using boost::accumulators::histogram_min_range;
    using boost::accumulators::histogram_max_range;

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////////
        // The histogram histogram estimator returns a histogram of the sample
        // distribution. The positions and sizes of the bins are determined using a
        // specifiable number of cached samples (cache_size). The range between the
        // minimum and the maximum of the cached samples is subdivided into a
        // specifiable number of bins (num_bins) of same size. Additionally,
        // an under- and an overflow bin is added to capture future under- and
        // overflow samples. Once the bins are determined, the cached samples and
        // all subsequent samples are added to the correct bins. At the end, a
        // range of std::pair is return, where each pair contains the position of
        // the bin (lower bound) and the samples count (normalized with the
        // total number of samples).
        template <typename Sample>
        struct histogram_impl : boost::accumulators::accumulator_base
        {
#if BOOST_VERSION > 105400
            typedef typename boost::numeric::functional::fdiv_base<
                    Sample, std::size_t
                >::result_type float_type;
#else
            typedef typename boost::numeric::functional::average<
                    Sample, std::size_t
                >::result_type float_type;
#endif
            typedef std::vector<std::pair<float_type, float_type> > histogram_type;
            typedef std::vector<float_type> array_type;

            // for boost::result_of
            typedef boost::iterator_range<
                    typename histogram_type::iterator
                > result_type;

// conversion from 'const __int64' to 'const double', possible loss of data
#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable: 4244)
#endif
            template <typename Args>
            histogram_impl(Args const& args)
             :  num_bins_(args[histogram_num_bins]),
                minimum_(args[histogram_min_range]),
                maximum_(args[histogram_max_range]),
                bin_size_ (boost::numeric::average(
                    args[histogram_max_range] - args[histogram_min_range],
                    args[histogram_num_bins])
                ),
                samples_in_bin_(std::size_t(args[histogram_num_bins] + 2), 0.),
                bin_positions_(args[histogram_num_bins] + 2),
                histogram_(
                   std::size_t(args[histogram_num_bins] + 2),
                   std::make_pair(0, 1)
                ),
                is_dirty_(true)
            {
                // determine bin positions (their lower bounds)
                for (std::size_t i = 0; i < this->num_bins_ + 2; ++i)
                {
                    this->bin_positions_[i] = minimum_ + (i - 1.0) * bin_size_;
                }
            }
#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif

            template <typename Args>
            void operator()(Args const &args)
            {
                //std::size_t cnt = count(args);
                {
                    if (args[boost::accumulators::sample] <
                        this->bin_positions_[1])
                    {
                        ++(this->samples_in_bin_[0]);
                    }
                    else if (args[boost::accumulators::sample] >=
                        this->bin_positions_[this->num_bins_ + 1])
                    {
                        ++(this->samples_in_bin_[this->num_bins_ + 1]);
                    }
                    else
                    {
                        typename array_type::iterator it = std::upper_bound(
                            this->bin_positions_.begin()
                          , this->bin_positions_.end()
                          , args[boost::accumulators::sample]
                        );

                        std::size_t d =
                            std::distance(this->bin_positions_.begin(), it);
                        ++(this->samples_in_bin_[d - 1]);
                    }
                }
            }

            template <typename Args>
            result_type result(Args const &args) const
            {
                {
                    // creates a vector of std::pair where each pair i holds
                    // the values bin_positions[i] (x-axis of histogram) and
                    // samples_in_bin[i] / cnt (y-axis of histogram).

                    for (std::size_t i = 0; i < this->num_bins_ + 2; ++i)
                    {
                        this->histogram_[i] = std::make_pair(
                            this->bin_positions_[i],
                            boost::numeric::average(
                                this->samples_in_bin_[i],
                                boost::accumulators::count(args)));
                    }
                }

                // returns a range of pairs
                return boost::make_iterator_range(this->histogram_);
            }

        private:
            std::size_t num_bins_;        // number of bins
            float_type minimum_;
            float_type maximum_;
            float_type bin_size_;
            array_type samples_in_bin_;  // number of samples in each bin
            array_type bin_positions_;   // lower bounds of bins
            mutable histogram_type histogram_;      // histogram
            mutable bool is_dirty_;
        };
    }
}}

///////////////////////////////////////////////////////////////////////////////
// tag::histogram
namespace boost { namespace accumulators
{
    namespace tag
    {
        struct histogram
          : depends_on<count>,
            histogram_num_bins,
            histogram_min_range,
            histogram_max_range
        {
            struct impl
            {
                template<typename Sample, typename Weight>
                struct apply
                {
                    typedef hpx::util::detail::histogram_impl<Sample> type;
                };
            };
        };
    }

    ///////////////////////////////////////////////////////////////////////////////
    // extract::histogram
    namespace extract
    {
        extractor<tag::histogram> const histogram = {};
    }
}} // namespace boost::accumulators

namespace hpx { namespace util
{
    namespace tag
    {
        using boost::accumulators::tag::histogram;
    }

    using boost::accumulators::extract::histogram;
}}

#endif

