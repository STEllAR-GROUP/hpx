//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_MOVE)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_MOVE

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/parallel/segmented_algorithms/movecopy.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <iostream>

#include <boost/type_traits/is_same.hpp>

namespace hpx {
    namespace parallel {
        HPX_INLINE_NAMESPACE(v1) {
            ///////////////////////////////////////////////////////////////////////////
            // segmented_move
            namespace detail {
            }
        }
    }
}

#endif
