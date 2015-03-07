//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHMS_SET_OPERATION_MAR_06_2015_0704PM)
#define HPX_PARALLEL_ALGORITHMS_SET_OPERATION_MAR_06_2015_0704PM

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/algorithm_result.hpp>

#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_scalar.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    template <typename OutIter>
    struct set_operations_buffer
    {
        template <typename T>
        class rewritable_ref
        {
        public:
            rewritable_ref() : item_(0) {}
            rewritable_ref(T const& item) : item_(item) {}

            rewritable_ref& operator= (T const& item)
            {
                item_ = &item;
                return *this;
            }

            operator T const&() const
            {
                HPX_ASSERT(item_ != 0);
                return *item_;
            }

        private:
            T const* item_;
        };

        typedef typename std::iterator_traits<OutIter>::value_type value_type;
        typedef typename boost::mpl::if_<
            boost::is_scalar<value_type>, value_type, rewritable_ref<value_type>
        >::type type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename RanIter1, typename RanIter2,
        typename OutIter, typename F, typename Combiner, typename SetOp>
    typename algorithm_result<ExPolicy, OutIter>::type
    set_operation(ExPolicy const& policy,
        RanIter1 first1, RanIter1 last1, RanIter2 first2, RanIter2 last2,
        OutIter dest, F && f, Combiner && combiner, SetOp && setop)
    {
        typedef algorithm_result<ExPolicy, OutIter> result;
        typedef typename std::iterator_traits<RanIter1>::difference_type
            difference_type1;
        typedef typename std::iterator_traits<RanIter2>::difference_type
            difference_type2;

        // allocate intermediate buffer
        difference_type1 len1 = std::distance(first1, last1);
        difference_type2 len2 = std::distance(first2, last2);

        typedef typename set_operations_buffer<OutIter>::type buffer_type;

        boost::shared_array<buffer_type> buffer(
            new buffer_type[combiner(len1, len2)]);

        // fill the buffer piecewise

        // accumulate real length

        // finally, copy data to destination

        return result::get(std::move(dest));
    }

    /// \endcond
}}}}

#endif


