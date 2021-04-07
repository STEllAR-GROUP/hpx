//  Copyright (c) 2021 @rainmaker6
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/shift_left.hpp
#pragma once
#include<hpx/iterator_support/zip_iterator.hpp>

namespace hpx {
    namespace parallel {
        namespace algorithms {
            template<typename T>
            using difference_type_t = typename hpx::iterators<T>::difference_type;
            template<typename T>
            using iterator_category_t = typename hpx::iterators<T>::iterator_category;
            template<class T, class Tag, class = void>
            constexpr bool is_category = false;
            template<class T>
            constexpr difference_type_t<T> bounded_advance(T& i, difference_type_t<T>& n, T& const bound) {
                if constexpr (is_category<T, hpx::bidirectional_traversal_tag>) {
                    for (; n < 0 && i != bound; ++n, void(--i)) {
                        ;
                    }                
                }
                for (; n > 0 && i != bound; --n, void(++i)) {
                    ;                
                }
                return n;
            }
            template<class FwdIter>
            FwdIter shift_right(FwdIter& first, FwdIter& last, difference_type_t<FwdIter>& n) {
                if (n <= 0) {
                    return last;
                }
                auto mid = first;
                if (::bounded_advance(mid, n, last)) {
                    return first;
                }
                return hpx::move(hpx::move(mid), hpx::move(last), hpx::move(first));
            }
        }
    }
}