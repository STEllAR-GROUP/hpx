//  Copyright (c) 2019 Jan Melech
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_LEXICOGRAPHICAL_COMPARE)
#define HPX_PARALLEL_DETAIL_LEXICOGRAPHICAL_COMPARE

#include <hpx/config.hpp>

#include <functional>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail
{
    // provide implementation of std::lexicographical_compare
    // supporting iterators/sentinels
    template <typename InIter1B, typename InIter1E, typename InIter2B,
            typename InIter2E, typename Pred>
    inline bool lexicographical_compare_(InIter1B first1, InIter1E last1,
            InIter2B first2, InIter2E last2, Pred pred)
    {
        for ( ; (first1 != last1) && (first2 != last2); ++first1, (void) ++first2 ) {
            if (pred(*first1, *first2)) return true;
            if (pred(*first2, *first1)) return false;
        }
        return (first1 == last1) && (first2 != last2);
    }
}}}}

#endif
