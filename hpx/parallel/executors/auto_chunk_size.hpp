//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/auto_chunk_size.hpp

#if !defined(HPX_PARALLEL_AUTO_CHUNK_SIZE_JUL_31_2015_0742PM)
#define HPX_PARALLEL_AUTO_CHUNK_SIZE_JUL_31_2015_0742PM

#include <hpx/config.hpp>
#include <hpx/traits/is_executor_parameters.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/date_time_chrono.hpp>

#include <cstddef>
#include <algorithm>

#include <boost/cstdint.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    /// Loop iterations are divided into pieces and then assigned to threads.
    /// The number of loop iterations combined is determined based on
    /// measurements of how long the execution of 1% of the overall number of
    /// iterations takes.
    /// This executor parameters type makes sure that as many loop iterations
    /// are combined as necessary to run for the amount of time specified.
    ///
    struct auto_chunk_size : executor_parameters_tag
    {
    public:
        /// Construct an \a auto_chunk_size executor parameters object
        ///
        /// \note Default constructed \a auto_chunk_size executor parameter
        ///       types will use 80 microseconds as the minimal time for which
        ///       any of the scheduled chunks should run.
        ///
        auto_chunk_size()
          : min_time_(80000)
        {}

        /// Construct an \a auto_chunk_size executor parameters object
        ///
        /// \param rel_time     [in] The time duration to use as the minimum
        ///                     to decide how many loop iterations should be
        ///                     combined.
        ///
        explicit auto_chunk_size(util::steady_duration const& rel_time)
          : min_time_(rel_time.value().count())
        {}

        /// \cond NOINTERNAL
        // Estimate a chunk size based on number of cores used.
        template <typename Executor, typename F>
        std::size_t get_chunk_size(Executor& exec, F && f, std::size_t count)
        {
            std::size_t const cores =
                executor_traits<Executor>::os_thread_count(exec);

            if (count > 100*cores)
            {
                using hpx::util::high_resolution_clock;
                boost::uint64_t t = high_resolution_clock::now();

                std::size_t test_chunk_size = f();
                if (test_chunk_size != 0)
                {
                    t = (high_resolution_clock::now() - t) / test_chunk_size;
                    if (t != 0)
                    {
                        // return chunk size which will create the required
                        // amount of work
                        return (std::min)(count, (std::size_t)(min_time_ / t));
                    }
                }
            }

            return (count + cores - 1) / cores;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & min_time_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        boost::uint64_t min_time_;      // nanoseconds
        /// \endcond
    };
}}}

#endif
