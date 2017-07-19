//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/partitioned_vector/partitioned_vector_view.hpp

#ifndef HPX_PARTITIONED_VECTOR_LOCAL_VIEW_HPP
#define HPX_PARTITIONED_VECTOR_LOCAL_VIEW_HPP

#include <hpx/components/containers/partitioned_vector/partitioned_vector_view.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_local_view_iterator.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
/// \cond NOINTERNAL

namespace hpx
{
    template<typename T, std::size_t N, typename Data>
    struct partitioned_vector_local_view
    : public hpx::partitioned_vector_view<T,N,Data>
    {
    private:
        using base_type = hpx::partitioned_vector_view<T,N,Data>;
        using base_iterator = typename base_type::iterator;

    public:
        using value_type = T;
        using iterator = typename
            hpx::partitioned_vector_local_view_iterator<Data, base_iterator>;

        explicit partitioned_vector_local_view(base_type const & global_pview )
        : base_type(global_pview)
        {}

    // Iterator interfaces
        iterator begin()
        {
            base_type & base(*this);
            return iterator(base.begin(), base.end());
        }

        iterator end()
        {
            base_type & base(*this);
            return iterator( base.end(), base.end() );
        }

    };

    template<typename T, std::size_t N, typename Data>
    partitioned_vector_local_view<T,N,Data>
    local_view(hpx::partitioned_vector_view<T,N,Data> const & base)
    {
        return partitioned_vector_local_view<T,N,Data>( base );
    }

}

#endif // PARTITIONED_VECTOR_LOCAL_VIEW_HPP
