//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/partitioned_vector/partitioned_vector_local_view_iterator.hpp

#ifndef PARTITIONED_VECTOR_LOCAL_VIEW_ITERATOR_HPP
#define PARTITIONED_VECTOR_LOCAL_VIEW_ITERATOR_HPP

#include <hpx/components/containers/partitioned_vector/detail/view_element.hpp>
#include <boost/iterator/iterator_adaptor.hpp>

namespace hpx {

    template <typename DataType, typename BaseIter>
    class partitioned_vector_local_view_iterator
      : public boost::iterator_adaptor<
            partitioned_vector_local_view_iterator<DataType, BaseIter>,
            BaseIter,
            DataType,
            std::forward_iterator_tag>
    {
    private:
        using base_type
            = boost::iterator_adaptor<
                partitioned_vector_local_view_iterator<DataType, BaseIter>,
                BaseIter,
                DataType,
                std::forward_iterator_tag>;

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


}

#endif
