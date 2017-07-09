//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/partitioned_vector/partitioned_vector_view_iterator.hpp

#if !defined(PARTITIONED_VECTOR_VIEW_ITERATOR_HPP)
#define PARTITIONED_VECTOR_VIEW_ITERATOR_HPP

#include <hpx/components/containers/partitioned_vector/detail/view_element.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_segmented_iterator.hpp>
#include <hpx/util/detail/pack.hpp>

#include <boost/iterator/iterator_facade.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <iterator>

namespace hpx {

    template<typename T, std::size_t N, typename Data>
    class partitioned_vector_view_iterator
    : public boost::iterator_facade<
                partitioned_vector_view_iterator<T,N,Data>,
                hpx::detail::view_element<T,Data>,
                std::random_access_iterator_tag,
                hpx::detail::view_element<T,Data> >
    {
    private:
        using pvector_iterator = hpx::vector_iterator<T,Data>;
        using segment_iterator = typename pvector_iterator::segment_iterator;
        using indices =
            typename hpx::util::detail::make_index_pack_unroll<N>::type;

    template<std::size_t... I>
    std::size_t  increment_solver( std::size_t dist,
        hpx::util::detail::pack_c<std::size_t, I...> ) const
    {
        std::size_t max = N-1;
        std::size_t offset = 0;
        std::size_t carry = dist;
        std::size_t tmp;

        // More expensive than a usual incrementation but did not find another
        // solution
        (void)std::initializer_list<int>
        { ( static_cast<void>(
                carry   -= tmp = (carry/sw_basis_[max-I]) * sw_basis_[max-I],
                offset  += (tmp/sw_basis_[max-I]) * hw_basis_[max-I]),
            0 )...
        };

        return offset;
    }

    public:
        using element_type = hpx::detail::view_element<T,Data>;

        explicit partitioned_vector_view_iterator(
              hpx::lcos::spmd_block const & block
            , segment_iterator const & begin
            , segment_iterator const & end
            , std::array<std::size_t, N+1> const & sw_basis
            , std::array<std::size_t, N+1> const & hw_basis
            , std::size_t count
            )
        : block_(block), t_(begin), begin_(begin), end_(end), count_(count)
        , sw_basis_(sw_basis), hw_basis_(hw_basis)
        {}

        partitioned_vector_view_iterator(
            partitioned_vector_view_iterator const &) = default;

        partitioned_vector_view_iterator(
            partitioned_vector_view_iterator &&) = default;

        // Note : partitioned_vector_view_iterator is not assignable
        // because it owns references members
        partitioned_vector_view_iterator
        operator=(partitioned_vector_view_iterator const &) = delete;

        partitioned_vector_view_iterator
        operator=(partitioned_vector_view_iterator &&) = delete;

    private:
        friend class boost::iterator_core_access;

        void increment()
        {
            std::size_t offset = increment_solver(++count_, indices() );
            t_ = begin_ + offset;
        }

        void decrement()
        {
            std::size_t offset = increment_solver(--count_, indices() );
            t_ = begin_ + offset;
        }

        void advance(std::size_t n)
        {
            std::size_t offset = increment_solver(count_+=n, indices() );
            t_ = begin_ + offset;
        }

        bool equal(partitioned_vector_view_iterator const& other) const
        {
            return this->count_ == other.count_;
        }

        // Will not return a datatype but a view_element type
        element_type dereference() const
        {
            return hpx::detail::view_element<T,Data>(block_,begin_,end_,t_);
        }

        std::ptrdiff_t distance_to(partitioned_vector_view_iterator const& other) const
        {
            return other.count_ - count_;
        }

        hpx::lcos::spmd_block const & block_;
        segment_iterator t_, begin_, end_;
        std::size_t count_;
        std::array< std::size_t, N+1 > const & sw_basis_;
        std::array< std::size_t, N+1 > const & hw_basis_;
    };
}

#endif
