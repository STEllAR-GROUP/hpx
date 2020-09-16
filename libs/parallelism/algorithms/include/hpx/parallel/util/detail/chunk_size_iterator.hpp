//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/iterator_support/iterator_facade.hpp>
#include <hpx/statistics/min.hpp>

#include <hpx/execution/algorithms/detail/predicates.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    struct chunk_size_iterator
      : public hpx::util::iterator_facade<chunk_size_iterator<Iterator>,
            hpx::tuple<Iterator, std::size_t> const, std::input_iterator_tag>
    {
    private:
        typedef hpx::util::iterator_facade<chunk_size_iterator<Iterator>,
            hpx::tuple<Iterator, std::size_t> const, std::input_iterator_tag>
            base_type;

    public:
        HPX_HOST_DEVICE
        chunk_size_iterator(
            Iterator it, std::size_t chunk_size, std::size_t count = 0)
          : data_(it, (hpx::util::min)(chunk_size, count))
          , chunk_size_(chunk_size)
          , count_(count)
        {
        }

    private:
        HPX_HOST_DEVICE
        Iterator& iterator()
        {
            return hpx::get<0>(data_);
        }
        HPX_HOST_DEVICE
        Iterator iterator() const
        {
            return hpx::get<0>(data_);
        }

        HPX_HOST_DEVICE
        std::size_t& chunk()
        {
            return hpx::get<1>(data_);
        }
        HPX_HOST_DEVICE
        std::size_t chunk() const
        {
            return hpx::get<1>(data_);
        }

    protected:
        friend class hpx::util::iterator_core_access;

        HPX_HOST_DEVICE bool equal(chunk_size_iterator const& other) const
        {
            return iterator() == other.iterator() && count_ == other.count_ &&
                chunk_size_ == other.chunk_size_;
        }

        HPX_HOST_DEVICE typename base_type::reference dereference() const
        {
            return data_;
        }

        HPX_HOST_DEVICE void increment()
        {
            // prepare next value
            count_ -= chunk();

            iterator() = parallel::v1::detail::next(iterator(), chunk());
            chunk() = (hpx::util::min)(chunk_size_, count_);
        }

    private:
        hpx::tuple<Iterator, std::size_t> data_;
        std::size_t chunk_size_;
        std::size_t count_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    struct chunk_size_idx_iterator
      : public hpx::util::iterator_facade<chunk_size_idx_iterator<Iterator>,
            hpx::tuple<Iterator, std::size_t, std::size_t> const,
            std::input_iterator_tag>
    {
    private:
        typedef hpx::util::iterator_facade<chunk_size_idx_iterator<Iterator>,
            hpx::tuple<Iterator, std::size_t, std::size_t> const,
            std::input_iterator_tag>
            base_type;

    public:
        HPX_HOST_DEVICE
        chunk_size_idx_iterator(Iterator it, std::size_t chunk_size,
            std::size_t count = 0, std::size_t base_idx = 0)
          : data_(it, (hpx::util::min)(chunk_size, count), base_idx)
          , count_(count)
          , chunk_size_(chunk_size)
        {
        }

    private:
        HPX_HOST_DEVICE
        Iterator& iterator()
        {
            return hpx::get<0>(data_);
        }
        HPX_HOST_DEVICE
        Iterator iterator() const
        {
            return hpx::get<0>(data_);
        }

        HPX_HOST_DEVICE
        std::size_t& chunk()
        {
            return hpx::get<1>(data_);
        }
        HPX_HOST_DEVICE
        std::size_t chunk() const
        {
            return hpx::get<1>(data_);
        }

        HPX_HOST_DEVICE
        std::size_t& base_index()
        {
            return hpx::get<2>(data_);
        }

    protected:
        friend class hpx::util::iterator_core_access;

        HPX_HOST_DEVICE bool equal(chunk_size_idx_iterator const& other) const
        {
            return iterator() == other.iterator() && count_ == other.count_ &&
                chunk_size_ == other.chunk_size_;
        }

        HPX_HOST_DEVICE typename base_type::reference dereference() const
        {
            return data_;
        }

        HPX_HOST_DEVICE void increment()
        {
            // prepare next value
            count_ -= chunk();

            iterator() = parallel::v1::detail::next(iterator(), chunk());
            base_index() += chunk();
            chunk() = (hpx::util::min)(chunk_size_, count_);
        }

    private:
        hpx::tuple<Iterator, std::size_t, std::size_t> data_;
        std::size_t count_;
        std::size_t chunk_size_;
    };
}}}}    // namespace hpx::parallel::util::detail
