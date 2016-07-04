//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_DETAIL_CHUNK_SIZE_ITERATOR_JUL_03_2016_0949PM)
#define HPX_PARALLEL_UTIL_DETAIL_CHUNK_SIZE_ITERATOR_JUL_03_2016_0949PM

#include <hpx/config.hpp>
#include <hpx/util/tuple.hpp>

#include <hpx/parallel/algorithms/detail/predicates.hpp>

#include <boost/iterator/iterator_facade.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    struct chunk_size_iterator
      : public boost::iterator_facade<
            chunk_size_iterator<Iterator>,
            hpx::util::tuple<Iterator, std::size_t> const,
            std::forward_iterator_tag>
    {
    private:
        typedef boost::iterator_facade<
                chunk_size_iterator<Iterator>,
                hpx::util::tuple<Iterator, std::size_t> const,
                std::forward_iterator_tag
            > base_type;

    public:
        chunk_size_iterator(Iterator it, std::size_t chunk_size,
                std::size_t count = 0)
          : data_(it, (std::min)(chunk_size, count))
          , chunk_size_(chunk_size)
          , count_(count)
        {}

    private:
        Iterator& iterator() { return hpx::util::get<0>(data_); }
        Iterator iterator() const { return hpx::util::get<0>(data_); }

        std::size_t& chunk() { return hpx::util::get<1>(data_); }
        std::size_t chunk() const { return hpx::util::get<1>(data_); }

    protected:
        friend class boost::iterator_core_access;

        bool equal(chunk_size_iterator const& other) const
        {
            return iterator() == other.iterator() &&
                count_ == other.count_ &&
                chunk_size_ == other.chunk_size_;
        }

        typename base_type::reference dereference() const
        {
            return data_;
        }

        void increment()
        {
            // prepare next value
            count_ -= chunk();

            iterator() =  parallel::v1::detail::next(iterator(), chunk());
            chunk() = (std::min)(chunk_size_, count_);
        }

    private:
        hpx::util::tuple<Iterator, std::size_t> data_;
        std::size_t chunk_size_;
        std::size_t count_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    struct chunk_size_idx_iterator
      : public boost::iterator_facade<
            chunk_size_idx_iterator<Iterator>,
            hpx::util::tuple<Iterator, std::size_t, std::size_t> const,
            std::forward_iterator_tag>
    {
    private:
        typedef boost::iterator_facade<
                chunk_size_idx_iterator<Iterator>,
                hpx::util::tuple<Iterator, std::size_t, std::size_t> const,
                std::forward_iterator_tag
            > base_type;

    public:
        chunk_size_idx_iterator(Iterator it, std::size_t chunk_size,
                std::size_t count = 0, std::size_t base_idx = 0)
          : data_(it, (std::min)(chunk_size, count), base_idx)
          , count_(count)
          , chunk_size_(chunk_size)
        {}

    private:
        Iterator& iterator() { return hpx::util::get<0>(data_); }
        Iterator iterator() const { return hpx::util::get<0>(data_); }

        std::size_t& chunk() { return hpx::util::get<1>(data_); }
        std::size_t chunk() const { return hpx::util::get<1>(data_); }

        std::size_t& base_index() { return hpx::util::get<2>(data_); }

    protected:
        friend class boost::iterator_core_access;

        bool equal(chunk_size_idx_iterator const& other) const
        {
            return iterator() == other.iterator() &&
                count_ == other.count_ &&
                chunk_size_ == other.chunk_size_;
        }

        typename base_type::reference dereference() const
        {
            return data_;
        }

        void increment()
        {
            // prepare next value
            count_ -= chunk();

            iterator() = parallel::v1::detail::next(iterator(), chunk());
            base_index() += chunk();
            chunk() = (std::min)(chunk_size_, count_);
        }

    private:
        hpx::util::tuple<Iterator, std::size_t, std::size_t> data_;
        std::size_t count_;
        std::size_t chunk_size_;
    };
}}}}

#endif

