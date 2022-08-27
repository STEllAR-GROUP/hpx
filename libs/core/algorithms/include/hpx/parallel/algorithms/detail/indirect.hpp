//  Copyright (c) 2015-2017 Francisco Jose Tapia
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>

#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    /// \class less_ptr_no_null
    ///
    /// \remarks this is the comparison object for pointers. Receive a object
    ///          for to compare the objects pointed. The pointers can't be
    ///          nullptr
    template <typename Iter, typename Sent,
        typename Comp =
            std::less<typename std::iterator_traits<Iter>::value_type>>
    struct less_ptr_no_null
    {
        Comp comp;

        inline less_ptr_no_null(Comp C1 = Comp())
          : comp(HPX_MOVE(C1))
        {
        }

        inline bool operator()(Iter T1, Sent T2) const
        {
            return comp(*T1, *T2);
        }
    };

    /// \brief Create a index of iterators to the elements
    /// \tparam Iter : iterator to store in the index vector
    /// \param [in] first : iterator to the first element of the range
    /// \param [in] last : iterator to the element after the last of the range
    /// \param [in/out] v_iter : vector where store the iterators of the index
    template <typename Iter, typename Sent>
    void create_index(Iter first, Sent last, std::vector<Iter>& v_iter)
    {
        auto nelem = detail::distance(first, last);
        HPX_ASSERT(nelem >= 0);
        v_iter.clear();
        v_iter.reserve(nelem);
        for (/**/; first != last; ++first)
        {
            v_iter.push_back(first);
        }
    }

    /// \brief sort the elements according of the sort of the index
    /// \tparam Iter : iterators of the index
    /// \param [in] first : iterator to the first element of the data
    /// \param [in] v_iter : vector sorted of the iterators

    template <typename Iter>
    void sort_index(Iter first, std::vector<Iter>& v_iter)
    {
        using value_type = typename std::iterator_traits<Iter>::value_type;

        std::size_t pos_dest = 0, pos_src = 0, pos_in_vector = 0;
        std::size_t nelem = v_iter.size();

        while (pos_in_vector < nelem)
        {
            while (pos_in_vector < nelem &&
                std::size_t(detail::distance(first, v_iter[pos_in_vector])) ==
                    pos_in_vector)
            {
                ++pos_in_vector;
            }

            if (pos_in_vector == nelem)
            {
                return;
            }

            pos_dest = pos_src = pos_in_vector;
            Iter it_dest = std::next(first, pos_dest);
            value_type Aux = HPX_MOVE(*it_dest);

            while ((pos_src = std::size_t(detail::distance(
                        first, v_iter[pos_dest]))) != pos_in_vector)
            {
                v_iter[pos_dest] = it_dest;
                Iter it_src = std::next(first, pos_src);
                *it_dest = HPX_MOVE(*it_src);
                it_dest = it_src;
                pos_dest = pos_src;
            }

            *it_dest = HPX_MOVE(Aux);
            v_iter[pos_dest] = it_dest;
            ++pos_in_vector;
        }
    }
}}}}    // namespace hpx::parallel::v1::detail
