//  Copyright (c) 2017 Zahra Khatami
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/adaptive_chunk_size.hpp

#if !defined(HPX_PARALLEL_ADAPTIVE_CHUNK_SIZE_FEB_13_2017)
#define HPX_PARALLEL_ADAPTIVE_CHUNK_SIZE_FEB_13_2017

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_executor_parameters.hpp>

#include <cstddef>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    /// Lerning model can be implemented on fo_each by using this chunk size
    /// as param

    struct adaptive_chunk_size : executor_parameters_tag
    {
        /// Construct a \a adaptive_chunk_size executor parameters object
        ///
        /// \param chunk_size   [in] The optional chunk size to use as the
        ///                     number of loop iterations to schedule together.
        ///                     The default chunk size is 1.
        ///
        HPX_CONSTEXPR explicit adaptive_chunk_size(std::size_t chunk_size = 1)
          : chunk_size_(chunk_size)
        {}

        /// \cond NOINTERNAL
        template <typename Executor, typename F>
        HPX_CONSTEXPR std::size_t
        get_chunk_size(Executor&, F &&, std::size_t, std::size_t)
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
