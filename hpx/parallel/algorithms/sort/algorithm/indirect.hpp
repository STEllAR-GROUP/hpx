//----------------------------------------------------------------------------
/// @file indirect.hpp
/// @brief Indirect algorithm
///
/// @author Copyright (c) 2010 Francisco José Tapia (fjtapia@gmail.com )\n
///         Distributed under the Boost Software License, Version 1.0.\n
///         ( See accompanyingfile LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __HPX_PARALLEL_SORT_ALGORITHM_INDIRECT_HPP
#define __HPX_PARALLEL_SORT_ALGORITHM_INDIRECT_HPP

#include <vector>
#include <type_traits>
#include <functional>
#include <iterator>
#include <hpx/parallel/algorithms/sort/tools/atomic.hpp>



namespace hpx		{
namespace parallel	{
HPX_INLINE_NAMESPACE(v2) { namespace boostsort		{
namespace algorithm	{

using std::iterator_traits ;
//
//##########################################################################
//                                                                        ##
//         S T R U C T     L E S S _ P T R _ N O _ N U L L                ##
//                                                                        ##
//##########################################################################
//
//---------------------------------------------------------------------------
/// @class less_ptr_no_null
///
/// @remarks this is the comparison object for pointers. Receive a object
///          for to compare the objects pointed. The pointers can't be nullptr
//---------------------------------------------------------------------------
template    <   class iter_t ,
                class comp_t
				=std::less<typename iterator_traits<iter_t>::value_type> >
struct less_ptr_no_null
{   //----------------------------- Variables -----------------------
    comp_t comp ;
    //----------------------------- Functions ----------------------
    inline less_ptr_no_null ( comp_t C1 = comp_t()):comp(C1){};
    inline bool operator ()( iter_t  T1,  iter_t  T2 ) const
    {   //-------------------- begin ------------------------------
        return  comp(*T1 ,*T2);
    };
};
//
//-----------------------------------------------------------------------------
//  function : create_index
/// @brief Create a index of iterators to the elements
/// @tparam iter_t : iterator to store in the index vector
/// @param [in] first : iterator to the first element of the range
/// @param [in] last : iterator to the element after the last of the range
/// @param [in/out] VP : vector where store the iterators of the index
//-----------------------------------------------------------------------------
template <class iter_t>
void create_index (iter_t first, iter_t last, std::vector<iter_t> &VP )
{   //----------------------------- begin -------------------------------
    auto N1 = last-first ;
    assert ( N1 >= 0 );
    VP.clear() ;
    VP.reserve ( N1);
    for ( ; first != last ; ++first) VP.push_back( first);
};
//
//-----------------------------------------------------------------------------
//  function : sort_index
/// @brief sort the elements according of the sort of the index
/// @tparam iter_t : iterators of the index
/// @param [in] first : iterator to the first element of the data
/// @param [in] VP : vector sorted of the iterators
//-----------------------------------------------------------------------------
template <class iter_t>
void sort_index (iter_t first, std::vector<iter_t> &VP )
{   //-------------------------- begin -------------------------------------
    typedef typename iterator_traits<iter_t>::value_type       value_t ;
    size_t Ax  = 0 , Bx =0 , Pos =0 , N = VP.size();
    iter_t itA, itB ;
    while ( Pos < N )
    {   while  ( Pos < N and (size_t (VP[Pos]-first)) == Pos ) ++Pos;
        if ( Pos == N ) return ;
        Ax = Bx = Pos ;
        itA = first + Ax  ;
        value_t Aux = std::move ( *itA);
        while ( (Bx = (size_t (VP[Ax]-first)))!= Pos  )
        {   VP[Ax] = itA;
            itB = first + Bx ;
            *itA = std::move ( *itB);
            itA = itB ;
            Ax = Bx ;
        };
        *itA = std::move ( Aux ) ;
        VP[Ax] = itA ;
        ++Pos ;
    };
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
