//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/partitioned_vector/partitioned_vector_local_view_iterator.hpp

#ifndef PARTITIONED_VECTOR_LOCAL_VIEW_ITERATOR_HPP
#define PARTITIONED_VECTOR_LOCAL_VIEW_ITERATOR_HPP

#include <hpx/components/containers/partitioned_vector/detail/view_element.hpp>
#include <boost/iterator/iterator_adaptor.hpp>

#include <utility>

namespace hpx {

    template <typename DataType, typename BaseIter>
    class partitioned_vector_local_view_iterator
      : public boost::iterator_adaptor<
            partitioned_vector_local_view_iterator<DataType, BaseIter>,
            BaseIter,
            DataType,
            std::forward_iterator_tag,
            DataType &>
    {
    private:
        using base_type
            = boost::iterator_adaptor<
                partitioned_vector_local_view_iterator<DataType, BaseIter>,
                BaseIter,
                DataType,
                std::forward_iterator_tag,
                DataType &>;

    public:
        partitioned_vector_local_view_iterator()
        {}

        explicit partitioned_vector_local_view_iterator(
            BaseIter && it, BaseIter && end)
        : base_type( std::forward<BaseIter>(it) ),
          end_( std::forward<BaseIter>(end) )
        {
            satisfy_predicate();
        }

        bool is_at_end() const
        {
            return this->base_reference() == end_;
        }

    private:
        friend class boost::iterator_core_access;

        template <typename, typename>
        friend class const_partitioned_vector_local_view_iterator;

        DataType & dereference() const
        {
            HPX_ASSERT(!is_at_end());
            return this->base_reference()->data();
        }

        void increment()
        {
            ++(this->base_reference());
            satisfy_predicate();
        }

        void satisfy_predicate()
        {
            while(this->base_reference() != end_ &&
                !this->base_reference()-> is_owned_by_current_thread())
            {
                ++( this->base_reference() );
            }
        }

    private:
        BaseIter end_;
    };

    template <typename DataType, typename BaseIter>
    class const_partitioned_vector_local_view_iterator
      : public boost::iterator_adaptor<
            const_partitioned_vector_local_view_iterator<DataType, BaseIter>,
            BaseIter,
            DataType,
            std::forward_iterator_tag,
            DataType const &>
    {
    private:
        using base_type
            = boost::iterator_adaptor<
                const_partitioned_vector_local_view_iterator<DataType, BaseIter>,
                BaseIter,
                DataType,
                std::forward_iterator_tag,
                DataType const &>;

    public:
        const_partitioned_vector_local_view_iterator()
        {}

        template<typename RightBaseIter>
        const_partitioned_vector_local_view_iterator(
            partitioned_vector_local_view_iterator<DataType,RightBaseIter>
                const & other)
            : base_type(other.base()), end_(other.end_)
        {}

        explicit const_partitioned_vector_local_view_iterator(
            BaseIter && it, BaseIter && end)
            : base_type( std::forward<BaseIter>(it) ),
                end_( std::forward<BaseIter>(end) )
        {
            satisfy_predicate();
        }

        bool is_at_end() const
        {
            return this->base_reference() == end_;
        }

    private:
        friend class boost::iterator_core_access;

        DataType const & dereference() const
        {
            HPX_ASSERT(!is_at_end());
            return this->base_reference()->data();
        }

        void increment()
        {
            ++(this->base_reference());
            satisfy_predicate();
        }

        void satisfy_predicate()
        {
            while(this->base_reference() != end_ &&
                !this->base_reference()-> is_owned_by_current_thread())
            {
                ++( this->base_reference() );
            }
        }

    private:
        BaseIter end_;
    };
}

#endif
