//----------------------------------------------------------------------------
/// @file sample_sort.hpp
/// @brief Sample Sort algorithm
///
/// @author Copyright (c) 2015 Francisco José Tapia (fjtapia@gmail.com )\n
///         Distributed under the Boost Software License, Version 1.0.\n
///         ( See accompanyingfile LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __HPX_PARALLEL_SORT_ALGORITHM_SAMPLE_SORT_HPP
#define __HPX_PARALLEL_SORT_ALGORITHM_SAMPLE_SORT_HPP

#include <functional>
#include <memory>
#include <type_traits>
#include <iterator>
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithms/sort/tools/atomic.hpp>
#include <hpx/parallel/algorithms/sort/tools/nthread.hpp>
#include <hpx/parallel/algorithms/sort/algorithm/spin_sort.hpp>
#include <hpx/parallel/algorithms/sort/util/range.hpp>
#include <hpx/parallel/algorithms/sort/util/merge_four.hpp>
#include <hpx/parallel/algorithms/sort/util/merge_vector.hpp>
#include <hpx/parallel/algorithms/sort/algorithm/indirect.hpp>

namespace hpx		{
namespace parallel 	{
HPX_INLINE_NAMESPACE(v2) { namespace boostsort		{
namespace algorithm	{

namespace su = util ;
using std::iterator_traits ;
using hpx::parallel::v2::boostsort::tools::NThread ;
using hpx::parallel::v2::boostsort::tools::NThread_HW ;
using hpx::parallel::v2::boostsort::tools::atomic_add ;
using hpx::parallel::v2::boostsort::util::uninit_merge_level4 ;
using hpx::parallel::v2::boostsort::util::merge_vector4 ;
//
///---------------------------------------------------------------------------
/// @struct sample_sort_tag
/// @brief This a structure for to implement a sample sort, exception
///        safe
/// @tparam
/// @remarks
//----------------------------------------------------------------------------
template < class iter_t,
           typename compare
           =std::less<typename iterator_traits<iter_t>::value_type >   >
struct sample_sort_tag
{
//------------------------------------------------------------------------
//                     DEFINITIONS
//------------------------------------------------------------------------
typedef typename iterator_traits<iter_t>::value_type   value_t ;
typedef range <iter_t>                      range_it ;
typedef range <value_t*>                    range_buf ;

//------------------------------------------------------------------------
//                VARIABLES AND CONSTANTS
//------------------------------------------------------------------------
static const uint32_t Thread_min = (1<<12) ;
uint32_t 	NThr, Ninterval ;
bool 		construct = false, owner = false  ;
compare 	comp ;
range_it 	global_range ;
range_buf 	global_buf ;


std::vector <std::vector <range_it > > 	VMem ;
std::vector <std::vector <range_buf> > 	VBuf ;
std::vector <range_it>                  VMIni ;
std::vector <range_buf>                 VBIni ;
std::atomic<uint32_t> 					NJobs ;


//----------------------------------------------------------------------------
//                       FUNCTIONS OF THE STRUCT
//----------------------------------------------------------------------------
void initial_configuration ( void);

//-----------------------------------------------------------------------------
//                 CONSTRUCTOR AND DESTRUCTOR
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
//  function : sample_sort_tag
/// @brief constructor of the class
///
/// @param [in] R : range of objects to sort
/// @param [in] comp : object for to compare two elements
/// @param [in] NT : NThread object for to define the number of threads to use
///                  in the process. By default is the number of thread HW
//-----------------------------------------------------------------------------
sample_sort_tag ( range_it R , compare cmp, NThread  NT);
//
//-----------------------------------------------------------------------------
//  function : sample_sort_tag
/// @brief constructor of the class
///
/// @param [in] R : range of elements to sort
/// @param [in] comp : object for to compare two elements
/// @param [in] NT : NThread object for to define the number of threads to use
///                  in the process. By default is the number of thread HW
/// @param [in] RB : range used as auxiliary memory
//-----------------------------------------------------------------------------
sample_sort_tag ( range_it R, compare cmp, NThread NT,range_buf RB);
//
//-----------------------------------------------------------------------------
//  function :~sample_sort_tag
/// @brief destructor of the class. The utility is to destroy the temporary
///        buffer used in the sorting process
//-----------------------------------------------------------------------------
~sample_sort_tag ( void);

//
//-----------------------------------------------------------------------------
//  function : execute first
/// @brief this a function to assign to each thread in the first merge
//-----------------------------------------------------------------------------
inline void execute_first ( void)
{   //------------------------------- begin ----------------------------------
    uint32_t Job =0 ;
    while ((Job = atomic_add(NJobs, 1)) < Ninterval   )
    {   uninit_merge_level4( VBIni[Job] , VMem[Job],VBuf[Job] ,comp);
    };
};
//
//-----------------------------------------------------------------------------
//  function : execute
/// @brief this is a function to assignt each thread the final merge
//-----------------------------------------------------------------------------
inline void execute ( void)
{   //------------------------------- begin ----------------------------------
    uint32_t Job =0 ;
    while ((Job = atomic_add(NJobs, 1)) < Ninterval   )
    {   merge_vector4 ( VBIni[Job] , VMIni[Job] ,VBuf[Job], VMem[Job], comp);

    };
};
//
//-----------------------------------------------------------------------------
//  function : first merge
/// @brief Implement the merge of the initially sparse ranges
//-----------------------------------------------------------------------------
inline void first_merge ( void)
{   //---------------------------------- begin -------------------------------
    NJobs =0 ;

	std::vector <hpx::future <void> > F ( NThr ) ;

	for ( uint32_t i =0 ; i < NThr ; ++i)
		F[i] = hpx::async (&sample_sort_tag<iter_t,compare>::execute_first , this);

	for ( uint32_t i =0 ; i < NThr ; ++i) F[i].get() ;

};
//
//-----------------------------------------------------------------------------
//  function : final merge
/// @brief Implement the final merge of the ranges
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
inline void final_merge ( void)
{   //---------------------------------- begin -------------------------------
    NJobs =0 ;
	std::vector <hpx::future <void> > F ( NThr ) ;
	for ( uint32_t i =0 ; i < NThr ; ++i)
		F[i] = hpx::async (&sample_sort_tag<iter_t,compare>::execute , this);

	for ( uint32_t i =0 ; i < NThr ; ++i) F[i].get() ;

};
//
//----------------------------------------------------------------------------
};//                    End class sample_sort_tag
//----------------------------------------------------------------------------
//
//
//############################################################################
//                                                                          ##
//              N O N    I N L I N E      F U N C T I O N S                 ##
//                                                                          ##
//                      O F   T H E      C L A S S                          ##
//                                                                          ##
//                   S A M P L E _ S O R T _ T A G                          ##
//                                                                          ##
//############################################################################
//-----------------------------------------------------------------------------
//  function : sample_sort_tag
/// @brief constructor of the class
///
/// @param [in] R : range of objects to sort
/// @param [in] comp : object for to compare two elements
/// @param [in] NT : NThread object for to define the number of threads to use
///                  in the process. By default is the number of thread HW
//-----------------------------------------------------------------------------
template <class iter_t, typename compare>
sample_sort_tag<iter_t,compare>:: sample_sort_tag (range_it R ,compare cmp,
		                                           NThread NT)
                                                  :owner(false),comp(cmp),
												   global_range(R),
				                                   global_buf (nullptr,nullptr)
{   //-------------------------- begin -------------------------------------

	assert ( R.valid());
    size_t NElem = R.size();
    construct= false ;
    NThr = NT() ;
    Ninterval = ( NThr <<4);
    NJobs = 0 ;

    if ( NT() <2 or NElem <= ( Thread_min))
    {   spin_sort (R.first, R.last, comp);
        return ;
    };

    //------------------- check if sort --------------------------------------
    bool SW = true ;
    for ( iter_t it1 = R.first, it2 = R.first+1 ;
          it2 != R.last and (SW = not comp(*it2,*it1));it1 = it2++);
    if (SW) return ;

    value_t * Ptr = std::get_temporary_buffer<value_t>(NElem).first ;
    if ( Ptr == nullptr) throw std::bad_alloc() ;
    owner = true ;
    global_buf = range_buf ( Ptr, Ptr + NElem);
    //------------------------------------------------------------------------
    //                    PROCESS
    //------------------------------------------------------------------------
    initial_configuration () ;

    first_merge ( );
    construct = true ;
    final_merge ( );

};
//
//-----------------------------------------------------------------------------
//  function : sample_sort_tag
/// @brief constructor of the class
///
/// @param [in] R : range of elements to sort
/// @param [in] comp : object for to compare two elements
/// @param [in] NT : NThread object for to define the number of threads to use
///                  in the process. By default is the number of thread HW
/// @param [in] RB : range used as auxiliary memory
//-----------------------------------------------------------------------------
template <class iter_t, typename compare>
sample_sort_tag<iter_t,compare>::sample_sort_tag ( range_it R, compare cmp,
		                                           NThread NT, range_buf RB)
                                                 : comp( cmp), global_range(R),
												   global_buf ( RB)
{   //-------------------------- begin -------------------------------------
	assert ( R.valid());
    size_t NElem = R.size();
    construct= false ;
    assert ( RB.first != nullptr and RB.size() >= NElem);
    NThr = NT() ;
    Ninterval = ( NThr <<3);
    NJobs = 0 ;

    if ( NT() <2 or NElem <= ( Thread_min))
    {   spin_sort_tag<iter_t, compare> (R, comp, RB);
        return ;
    };

    //------------------- check if sort --------------------------------------
    bool SW = true ;
    for ( iter_t it1 = R.first, it2 = R.first+1 ;
          it2 != R.last and (SW = not comp(*it2,*it1));it1 = it2++);
    if (SW) return ;

    //------------------------------------------------------------------------
    //                    PROCESS
    //------------------------------------------------------------------------
    initial_configuration () ;
    first_merge ( );
    construct = true ;
    final_merge ( );

};
//
//-----------------------------------------------------------------------------
//  function :~sample_sort_tag
/// @brief destructor of the class. The utility is to destroy the temporary
///        buffer used in the sorting process
//-----------------------------------------------------------------------------
template <class iter_t, typename compare>
sample_sort_tag<iter_t,compare>::~sample_sort_tag ( void)
{   //----------------------------------- begin -------------------------
    if ( construct)
    {   destroy ( global_buf);
        construct = false ;
    }
    if ( global_buf.first != nullptr and owner )
    	std::return_temporary_buffer ( global_buf.first) ;
};
//
//-----------------------------------------------------------------------------
//  function : initial_configuration
/// @brief Create the internal data structures, and obtain the inital set of
///        ranges to merge
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template <class iter_t, typename compare>
void sample_sort_tag<iter_t,compare>::initial_configuration ( void)
{   //--------------------------- begin --------------------------------------
    std::vector <range_it> 	VMem_thread ;
    std::vector <range_buf> VBuf_thread   ;
    size_t NElem = global_range.size() ;
    //------------------------------------------------------------------------
    size_t cupo = (NElem + NThr -1) / NThr ;
    iter_t 		it_first 	= global_range.first ;
    value_t * 	buf_first 	= global_buf.first ;

    for ( uint32_t i =0 ; i < NThr-1 ; ++i, it_first += cupo, buf_first += cupo)
    {	VMem_thread.emplace_back ( it_first, it_first + cupo) ;
        VBuf_thread.emplace_back ( buf_first, buf_first + cupo );
    };
    VMem_thread.emplace_back ( it_first, global_range.last) ;
    VBuf_thread.emplace_back ( buf_first, global_buf.last );

    //------------------------------------------------------------------------
    // Sorting of the ranges
    //------------------------------------------------------------------------
	std::vector <hpx::future <void> > F ( NThr ) ;

	for ( uint32_t i =0 ; i < NThr ; ++i)
	{	F[i] = hpx::async (spin_sort<iter_t,value_t,compare>,
				            VMem_thread[i].first, VMem_thread[i].last,
							comp, VBuf_thread[i].first,VBuf_thread[i].size());
	};

	for ( uint32_t i =0 ; i < NThr ; ++i) F[i].get() ;

    //------------------------------------------------------------------------
    // Obtain the vector of milestones
    //------------------------------------------------------------------------
    std::vector<iter_t> Vsample;
    Vsample.reserve ( NThr * (Ninterval-1)) ;

    for ( uint32_t i =0 ; i < NThr ; ++i)
    {   size_t distance = VMem_thread[i].size() / Ninterval ;
        for ( size_t j = 1 ,pos = distance; j < Ninterval; ++j,pos+=distance)
        {   Vsample.push_back (VMem_thread[i].first + pos );
        };
    };
    typedef less_ptr_no_null <iter_t, compare>  compare_ptr ;
    spin_sort  ( Vsample.begin() , Vsample.end(), compare_ptr(comp) );

    //------------------------------------------------------------------------
    // Create the final milestone vector
    //------------------------------------------------------------------------
    std::vector<iter_t> Vmilestone ;
    Vmilestone.reserve ( Ninterval);

    for ( uint32_t Pos =NThr >>1 ; Pos < Vsample.size() ; Pos += NThr )
        Vmilestone.push_back ( Vsample [ Pos]);

    //------------------------------------------------------------------------
    // Creation of the first vector of ranges
    //------------------------------------------------------------------------
    std::vector< std::vector<range <iter_t> > > VR  (NThr);

    for ( uint32_t i =0 ; i < NThr; ++i)
    {   iter_t itaux = VMem_thread[i].first ;
        for ( uint32_t k =0 ; k < (Ninterval -1) ; ++k)
        {   iter_t it2 = std::upper_bound ( itaux,
                                            VMem_thread[i].last ,
                                            * Vmilestone[k], comp );
            VR[i].emplace_back ( itaux, it2);
            itaux = it2 ;
        };
        VR[i].emplace_back(itaux,VMem_thread[i].last );
    };

    //------------------------------------------------------------------------
    // Copy in buffer and  creation of the final matrix of ranges
    //------------------------------------------------------------------------
    VMem.resize ( Ninterval);
    VBuf.resize ( Ninterval);
    VMIni.reserve (Ninterval);
    VBIni.reserve (Ninterval);

    for ( uint32_t i =0 ; i < Ninterval ; ++i)
    {   VMem[i].reserve ( NThr);
        VBuf[i].reserve ( NThr);
    };
    iter_t it = global_range.first ;
    value_t * it_buf = global_buf.first ;
    for ( uint32_t k =0 ; k < Ninterval ; ++k)
    {   size_t N =0 ;
        for ( uint32_t i = 0 ; i< NThr ; ++i)
        {   size_t N2 = VR[i][k].size();
            if ( N2 != 0 ) VMem[k].push_back(VR[i][k] );
            N += N2 ;
        };
        VMIni.emplace_back (it,it + N  );
        VBIni.emplace_back (it_buf , it_buf+ N) ;

        it += N ;
        it_buf += N ;
    };
};
//
//############################################################################
//                                                                          ##
//                 F U N C T I O N   F O R M A T                            ##
//                             A N D                                        ##
//               I N D I R E C T   F U N C T I O N S                        ##
//                                                                          ##
// These functions are for to select the correct format depending of the    ##
// number and type of the parameters                                        ##
//############################################################################
//
//-----------------------------------------------------------------------------
//  function : sample_sort
/// @brief envelope function for to call to a sample_sort_tag object
/// @tparam iter_t : iterator used for to access to the data
/// @param [in] first : iterator to the first element of the range
/// @param [in] last : iterator to next element after the last of the range
/// @param [in] NT : NThread object for to define the number of threads to use
///                  in the process. By default is the number of thread HW
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template < class iter_t >
void sample_sort ( iter_t first, iter_t last , const NThread &NT = NThread() )
{   //----------------------------------- begin ------------------------------
    typedef std::less<typename iterator_traits<iter_t>::value_type > compare;
    sample_sort_tag <iter_t,compare> ( range<iter_t>(first, last), compare(),NT);
};
//
//-----------------------------------------------------------------------------
//  function : sample_sort
/// @brief envelope function for to call to a sample_sort_tag object
/// @tparam iter_t : iterator used for to access to the data
/// @tparam compare : object for to compare two elements
/// @param [in] first : iterator to the first element of the range
/// @param [in] last : iterator to next element after the last of the range
/// @param [in] comp : object for to compare two elements
/// @param [in] NT : NThread object for to define the number of threads to use
///                  in the process. By default is the number of thread HW
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template < class iter_t,
          typename compare
          = std::less <typename iterator_traits<iter_t>::value_type>  >
void sample_sort ( iter_t first, iter_t last,
                   compare comp, const NThread &NT = NThread() )
{   //----------------------------- begin ----------------------------------
    sample_sort_tag<iter_t,compare> (range<iter_t>( first, last),comp,NT);
};


//############################################################################
//                                                                          ##
//                I N D I R E C T     F U N C T I O N S                     ##
//                                                                          ##
//############################################################################
//
//-----------------------------------------------------------------------------
//  function : indirect_sample_sort
/// @brief indirect sorting using the sample_sort algorithm
/// @tparam iter_t : iterator used for to access to the data
/// @param [in] first : iterator to the first element of the range
/// @param [in] last : iterator to next element after the last of the range
/// @param [in] NT : NThread object for to define the number of threads to use
///                  in the process. By default is the number of thread HW
//-----------------------------------------------------------------------------
template < class iter_t >
void indirect_sample_sort ( iter_t first, iter_t last ,
                                   NThread NT = NThread() )
{   //------------------------------- begin--------------------------
    typedef std::less <typename iterator_traits<iter_t>::value_type> compare ;
    typedef less_ptr_no_null <iter_t, compare>      compare_ptr ;

    std::vector<iter_t> VP ;
    create_index ( first , last , VP);
    sample_sort  ( VP.begin() , VP.end(), compare_ptr(),NT );
    sort_index ( first , VP) ;
};
//
//-----------------------------------------------------------------------------
//  function : indirect_sample_sort
/// @brief indirect sorting using the sample_sort algorithm
/// @tparam iter_t : iterator used for to access to the data
/// @tparam compare : object for to compare two elements
/// @param [in] first : iterator to the first element of the range
/// @param [in] last : iterator to next element after the last of the range
/// @param [in] comp : object for to compare two elements
/// @param [in] NT : NThread object for to define the number of threads to use
///                  in the process. By default is the number of thread HW
//-----------------------------------------------------------------------------
template < class iter_t,
          typename
          compare = std::less <typename iterator_traits<iter_t>::value_type> >
void indirect_sample_sort ( iter_t first, iter_t last,
                            compare comp1, NThread NT = NThread() )
{   //----------------------------- begin ----------------------------------
    typedef less_ptr_no_null <iter_t, compare>      compare_ptr ;

    std::vector<iter_t> VP ;
    create_index ( first , last , VP);
    sample_sort  ( VP.begin() , VP.end(), compare_ptr(comp1),NT );
    sort_index ( first , VP) ;
};
//
//****************************************************************************
};//    End namespace algorithm
};//    End namespace parallel
};};//    End HPX_INLINE_NAMESPACE(v2) 
};//    End namespace boost
//****************************************************************************
//
#endif
