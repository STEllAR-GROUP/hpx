//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/dynamic_chunk_size.hpp

#if !defined(HPX_PARALLEL_DYNAMIC_CHUNK_SIZE_AUG_01_2015_0234PM)
#define HPX_PARALLEL_DYNAMIC_CHUNK_SIZE_AUG_01_2015_0234PM

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_executor_parameters.hpp>

#include <cstddef>

namespace hpx { namespace parallel { inline namespace v3
{
    ///////////////////////////////////////////////////////////////////////////
    /// Loop iterations are divided into pieces of size \a chunk_size and then
    /// dynamically scheduled among the threads; when a thread finishes one
    /// chunk, it is dynamically assigned another If \a chunk_size is not
    /// specified, the default chunk size is 1.
    ///
    /// \note This executor parameters type is equivalent to OpenMP's DYNAMIC
    ///       scheduling directive.
    ///
    struct dynamic_chunk_size : executor_parameters_tag
    {
        /// Construct a \a dynamic_chunk_size executor parameters object
        ///
        /// \param chunk_size   [in] The optional chunk size to use as the
        ///                     number of loop iterations to schedule together.
        ///                     The default chunk size is 1.
        ///
        HPX_CONSTEXPR explicit dynamic_chunk_size(std::size_t chunk_size = 1)
          : chunk_size_(chunk_size)
        {}

        /// \cond NOINTERNAL
        template <typename Executor, typename F>
        HPX_CONSTEXPR std::size_t
        get_chunk_size(Executor&, F &&, std::size_t, std::size_t) const
        {
            return chunk_size_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & chunk_size_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        std::size_t chunk_size_;
        /// \endcond
    };
}}}

#endif
