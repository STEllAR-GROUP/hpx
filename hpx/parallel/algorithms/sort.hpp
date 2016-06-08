//  Copyright (c) 2015 John Biddiscombe
//  Copyright (c) 2015 Hartmut Kaiser
//  Copyright (c) 2015 Francisco Jose Tapia
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_SORT_OCT_2015)
#define HPX_PARALLEL_ALGORITHM_SORT_OCT_2015

#include <hpx/config.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <boost/exception_ptr.hpp>
#include <hpx/parallel/algorithms/sort/algorithm/blk_ind.hpp>
#include <hpx/parallel/algorithms/sort/algorithm/parallel_sort.hpp>
#include <hpx/parallel/algorithms/sort/algorithm/parallel_stable_sort.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <iostream>
#include <iterator>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{

//****************************************************************************
//             USING AND DEFINITIONS
//****************************************************************************
namespace hpx_algo 	= hpx::parallel::v2::boostsort::algorithm ;
namespace hpx_util 	= hpx::parallel::v2::boostsort::util ;
namespace hpx_tools = hpx::parallel::v2::boostsort::tools ;
namespace hpx_sort 	= hpx::parallel::v2::boostsort;

using std::iterator_traits ;
//----------------------------------------------------------------------------
// The code of the class NThread is in boost/sort/parallel/util/atomic.hpp
//----------------------------------------------------------------------------
using hpx_tools::NThread ;
using hpx_algo::less_ptr_no_null;
//
//-----------------------------------------------------------------------------
//  function : sample_sort
/// @brief this function implement a stable parallel sort with the algorithm of
///        sample sort. The number of threads to use is defined by the NThread
///        parameter
///
/// @param [in] parallel_execution_policy : This sorting is parallel
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] comp : object for to compare two elements pointed by iter_t
///                    iterators
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template <class iter_t,
          class compare= std::less < typename iterator_traits<iter_t>::value_type> >
void sample_sort (	hpx::parallel::parallel_execution_policy ,
            iter_t first, iter_t last , compare comp= compare()  )
{	NThread NT ( (uint32_t) hpx::get_os_thread_count()) ;
	hpx_algo::sample_sort ( first, last,comp, NT);
};
//
//-----------------------------------------------------------------------------
//  function : sort
/// @brief this function implement a non stable parallel sort. The number of
///        threads to use is defined by the NThread parameter
///
/// @param [in] parallel_execution_policy : This sorting is parallel
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] comp : object for to compare two elements pointed by iter_t
///                    iterators
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template <class iter_t,
          class compare= std::less < typename iterator_traits<iter_t>::value_type> >
void sort (	hpx::parallel::parallel_execution_policy ,
            iter_t first, iter_t last , compare comp= compare()  )
{	//----------------------------- begin -------------------------------
	NThread NT ( (uint32_t) hpx::get_os_thread_count()) ;
	if ( NT() >8 )
		hpx_algo::blk_ind ( first, last , comp, NT);
	else
        hpx_algo::parallel_sort ( first, last,comp);
};

//
//-----------------------------------------------------------------------------
//  function : sort
/// @brief this function implement a non stable sort, based internally in the
///        intro_sort algorithm. Run with 1 thread
///
/// @param [in] sequential_execution_policy : sort with 1 thread
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] comp : object for to compare two elements pointed by iter_t
///                    iterators
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template <class iter_t,
          class compare= std::less < typename iterator_traits<iter_t>::value_type> >
void sort (	hpx::parallel::sequential_execution_policy ,
            iter_t first, iter_t last ,compare comp= compare()  )
{	hpx_algo::intro_sort(first, last,comp);
};

//
//-----------------------------------------------------------------------------
//  function : stable_sort
/// @brief this function implement a stable parallel sort. The number of
///        threads to use is defined by the NThread parameter
///
/// @param [in] parallel_execution_policy : This sorting is parallel
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] comp : object for to compare two elements pointed by iter_t
///                    iterators
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template <class iter_t,
          class compare= std::less < typename iterator_traits<iter_t>::value_type> >
void stable_sort (	hpx::parallel::parallel_execution_policy ,
            iter_t first, iter_t last , compare comp= compare()  )
{	//--------------------------- begin --------------------------------------
	NThread NT ( (uint32_t) hpx::get_os_thread_count()) ;

	hpx_algo::parallel_stable_sort ( first, last,comp, NT);
};

//
//-----------------------------------------------------------------------------
//  function : stable_sort
/// @brief this function implement a stable sort, based internally in the new
///        smart_merge_sort algorithm. Run with 1 thread
///
/// @param [in] sequential_execution_policy : sort with 1 thread
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] comp : object for to compare two elements pointed by iter_t
///                    iterators
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template <class iter_t,
          class compare= std::less < typename iterator_traits<iter_t>::value_type> >
void stable_sort (	hpx::parallel::sequential_execution_policy ,
            iter_t first, iter_t last ,compare comp= compare()  )
{	hpx_algo::spin_sort(first, last,comp);
};
//
//****************************************************************************
};//    End HPX_INLINE_NAMESPACE(v2)
};//    End namespace parallel
};//    End namespace hpx
//****************************************************************************
//
#endif
