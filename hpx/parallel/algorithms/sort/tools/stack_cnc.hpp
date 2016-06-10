//----------------------------------------------------------------------------
/// @file   stack_cnc.hpp
/// @brief  This file contains the implementation of the several types of
///         recursive fastmutex for read and write
///
/// @author Copyright (c) 2010 2015 Francisco José Tapia (fjtapia@gmail.com )\n
///         Distributed under the Boost Software License, Version 1.0.\n
///         ( See accompanyingfile LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __TOOLS_STACK_CNC_HPP
#define __TOOLS_STACK_CNC_HPP

#include <hpx/parallel/algorithms/sort/tools/spinlock.hpp>
#include <vector>

namespace hpx
{
namespace parallel
{
HPX_INLINE_NAMESPACE(v2) { namespace boostsort
{
namespace tools
{

//###########################################################################
//                                                                         ##
//    ################################################################     ##
//    #                                                              #     ##
//    #                      C L A S S                               #     ##
//    #                   S T A C K _ C N C                          #     ##
//    #                                                              #     ##
//    ################################################################     ##
//                                                                         ##
//###########################################################################
//
//---------------------------------------------------------------------------
/// @class  stack_cnc
/// @brief This class is a concurrent stack controled by a spin_lock
/// @remarks
//---------------------------------------------------------------------------
template <typename T , typename Allocator=std::allocator<T> >
class stack_cnc
{
public:
//***************************************************************************
//                     D E F I N I T I O N S
//***************************************************************************
typedef std::vector<T,Allocator>                            vector_t ;
typedef typename vector_t::size_type                        size_type ;
typedef typename vector_t::difference_type                  difference_type ;
typedef typename vector_t::value_type                       value_type;
typedef typename vector_t::pointer                          pointer;
typedef typename vector_t::const_pointer                    const_pointer;
typedef typename vector_t::reference                        reference;
typedef typename vector_t::const_reference                  const_reference;
typedef typename vector_t::allocator_type                   allocator_type;
typedef Allocator                                           alloc_t ;
typedef hpx::util::spinlock  spinlock_t ;


protected:
//---------------------------------------------------------------------------
//                   Internal variables
//---------------------------------------------------------------------------
vector_t     V ;
mutable spinlock_t  spl;

public :
//
//***************************************************************************
//  C O N S T R U C T O R S     A N D    D E S T R U C T O R
//
//  explicit stack_cnc ( );
//  explicit stack_cnc ( const alloc_t &ALLC = alloc_t ())
//
//  stack_cnc ( const stack_cnc & VT )
//  stack_cnc ( stack_cnc && VT)
//
//  template < typename alloc_t2 =alloc_t , bool cnc2=cnc >
//  stack_cnc (const stack_cnc<value_type,cnc2 ,alloc_t2> &VT)
//
//  template < bool cnc2>
//  stack_cnc ( stack_cnc<value_type,cnc2,alloc_t> && VT)
//
//  virtual ~stack_cnc (void)
//
//***************************************************************************
//
//---------------------------------------------------------------------------
//  function : stack_cnc
/// @brief  constructor
//---------------------------------------------------------------------------
explicit inline stack_cnc (void ) : V() {}
//
//---------------------------------------------------------------------------
//  function : stack_cnc
/// @brief  constructor
/// @param [in] ALLC : Allocator
//---------------------------------------------------------------------------
explicit inline stack_cnc ( const Allocator &ALLC ):V(ALLC) {}
//
//---------------------------------------------------------------------------
//  function : stack_cnc
/// @brief  Copy constructor
/// @param [in] VT : stack_cnc from where copy the data
//---------------------------------------------------------------------------
stack_cnc ( const stack_cnc & VT ) = delete;
//
//---------------------------------------------------------------------------
//  function : stack_cnc
/// @brief  Copy constructor
/// @param [in] VT : stack_cnc from where copy the data
//---------------------------------------------------------------------------
template <typename Allocator2>
inline stack_cnc ( const std::vector<value_type,Allocator2> & VT ): V ( VT) {}
//
//---------------------------------------------------------------------------
//  function : stack_cnc
/// @brief  Move constructor
/// @param [in] VT : stack_cnc from where copy the data
//---------------------------------------------------------------------------
stack_cnc ( stack_cnc && VT) = delete;
//
//---------------------------------------------------------------------------
//  function : ~stack_cnc
/// @brief  Destructor
//---------------------------------------------------------------------------
virtual ~stack_cnc (void) {  V.clear(); }
//
//***************************************************************************
//  O P E R A T O R = , A S S I G N , C L E A R , S W A P
//
//  stack_cnc & operator= (const stack_cnc &VT)
//
//  template < typename alloc_t2 , bool cnc2>
//  stack_cnc & operator= (const stack_cnc<value_type,cnc2, alloc_t2> &VT)
//
//  template <bool cnc2>
//  stack_cnc & operator= ( stack_cnc<value_type,cnc2,alloc_t> &&A)
//
//  template <class InputIterator>
//  void assign ( InputIterator it_first, InputIterator it_last )
//
//  void assign ( unsize_type n, const value_type& u )
//
//  void clear(void)
//  void swap ( stack_cnc  & A ) HPX_NOEXCEPT
//
//***************************************************************************
//
//---------------------------------------------------------------------------
//  function : operator =
/// @brief Asignation operator
/// @param [in] VT : stack_cnc from where copy the data
/// @return Reference to the stack_cnc after the copy
//---------------------------------------------------------------------------
stack_cnc & operator= (const stack_cnc &VT) = delete;
//
//---------------------------------------------------------------------------
//  function : operator =
/// @brief Asignation operator
/// @param [in] VT : stack_cnc from where copy the data
/// @return Reference to the stack_cnc after the copy
//---------------------------------------------------------------------------
template < typename alloc_t2 >
stack_cnc & operator= (const std::vector<value_type,alloc_t2> &VT)
{
    //-------------------------- begin ------------------------------
    if ( this == &VT ) return *this ;
    std::lock_guard <spinlock_t>  S(spl);
    V = VT ;
    return *this ;
}
//
//---------------------------------------------------------------------------
//  function : clear
/// @brief Delete all the elements of the stack_cnc.
//---------------------------------------------------------------------------
void clear(void)
{
    std::lock_guard<spinlock_t>  S(spl);
    V.clear ( );
}
//
//---------------------------------------------------------------------------
//  function : swap
/// @brief swap the data between the two stack_cnc
/// @param [in] A : stack_cnc to swap
/// @return none
//---------------------------------------------------------------------------
void swap ( stack_cnc  & A ) HPX_NOEXCEPT
{
    //------------------ begin ------------------
    if ( this == &A ) return ;
    std::lock_guard<spinlock_t>  S(spl);
    V.swap( A.V);
}
//
//***************************************************************************
//  S I Z E , M A X _ S I Z E , R E S I Z E
//  C A P A C I T Y , E M P T Y , R E S E R V E
//
//  size_type size        ( void  ) const HPX_NOEXCEPT
//  size_type max_size    ( void  ) const HPX_NOEXCEPT
//  void      resize      ( unsize_type sz,value_type c = value_type())
//  size_type capacity    ( void  ) const HPX_NOEXCEPT
//  bool      empty       ( void  ) const HPX_NOEXCEPT
//  void      reserve     ( size_type n ) HPX_NOEXCEPT
//
//***************************************************************************
//
//---------------------------------------------------------------------------
//  function : size
/// @brief return the number of elements in the stack_cnc
/// @return number of elements in the stack_cnc
//---------------------------------------------------------------------------
size_type size ( void) const HPX_NOEXCEPT
{
    std::lock_guard<spinlock_t>  S(spl);
    return V.size() ;
}
//
//---------------------------------------------------------------------------
//  function :max_size
/// @brief return the maximun size of the container
/// @return maximun size of the container
//---------------------------------------------------------------------------
size_type max_size (void) const HPX_NOEXCEPT
{
    std::lock_guard<spinlock_t>  S(spl);
    return ( V.max_size() );
}
//
//---------------------------------------------------------------------------
//  function : shrink_to_fit
/// @brief resize the current vector size and change to sz.\n
///        If sz is smaller than the current size, delete elements to end\n
///        If sz is greater than the current size, insert elements to the
///        end with the value c
/// @param [in] sz : new size of the stack_cnc after the resize
/// @param [in] c : Value to insert if sz is greather than the current size
/// @return none
//---------------------------------------------------------------------------
void shrink_to_fit ( )
{
    std::lock_guard<spinlock_t>  S(spl);
    V.shrink_to_fit();
};
//
//---------------------------------------------------------------------------
//  function : capacity
/// @brief return the maximun size of the container
/// @return maximun size of the container
//---------------------------------------------------------------------------
size_type capacity ( void) const HPX_NOEXCEPT
{
    std::lock_guard<spinlock_t>  S(spl);
    return ( V.capacity() );
}
//
//---------------------------------------------------------------------------
//  function : empty
/// @brief indicate if the map is empty
/// @return true if the map is empty, false in any other case
//---------------------------------------------------------------------------
bool empty ( void) const HPX_NOEXCEPT
{
    std::lock_guard<spinlock_t>  S(spl);
    return (V.empty()) ;
}
//
//---------------------------------------------------------------------------
//  function : reserve
/// @brief Change the capacity for to contain , at least n elements
/// @param [in] n : number of elements for the new capacity
/// @return none
/// @remarks This function has not utility. It is provided only for
///          compatibility with the STL vector interface
//---------------------------------------------------------------------------
void reserve ( size_type n ) HPX_NOEXCEPT
{
    std::lock_guard<spinlock_t>  S(spl);
    V.reserve(n);
}
//
//---------------------------------------------------------------------------
//  function : copy
/// @brief explicit copy for to prevent the automatic copy with the operator =
/// @param [in] V : vector to copy
/// @return none
/// @remarks
//---------------------------------------------------------------------------
template< class Allocator2>
void copy ( std::vector<value_type,Allocator2> & V2)
{
    std::lock_guard<spinlock_t>  S(spl);
    V2 = V ;
}

//***************************************************************************
//          P U S H _ B A C K
//
//  template <class P >
//  iterator push_back (  P && D )
//
//  template <class P ,class Function>
//  iterator push_back_if ( P && D , Function && M )
//
//  template <class ... Args>
//  iterator emplace_back ( Args && ... args )
//
//  template <class Function , class ... Args>
//  iterator emplace_back_if ( Function && M , Args && ... args )
//
//***************************************************************************
//---------------------------------------------------------------------------
//  function : push_back
/// @brief Insert one element in the back of the container
/// @param [in] D : value to insert. Can ve a value, a reference or an rvalue
/// @return iterator to the element inserted
/// @remarks This operation is O ( const )
//---------------------------------------------------------------------------
void push_back (const value_type  & D )
{
    std::lock_guard<spinlock_t>  S(spl);
    V.push_back(D);
}

//---------------------------------------------------------------------------
//  function : emplace_back
/// @brief Insert one element in the back of the container
/// @param [in] args :group of arguments for to build the object to insert
/// @return iterator to the element inserted
/// @remarks This operation is O ( const )
//---------------------------------------------------------------------------
template <class ... Args>
void emplace_back ( Args && ... args )
{
    std::lock_guard<spinlock_t>  S(spl);
    V.emplace_back (std::forward <Args>(args) ...);
}
//---------------------------------------------------------------------------
//  function : push_back
/// @brief Insert one element in the back of the container
/// @param [in] D : value to insert. Can ve a value, a reference or an rvalue
/// @return iterator to the element inserted
/// @remarks This operation is O ( const )
//---------------------------------------------------------------------------
template <class Allocator2>
stack_cnc & push_back ( const std::vector<value_type,Allocator2> & V1)
{
    std::lock_guard<spinlock_t>  S(spl);
    for ( size_type i =0 ; i < V1.size() ; ++i)
        V.push_back(V1[i]);
    return *this ;
}
//
//***************************************************************************
//                  P O P _ B A C K
//
//  void pop_back ( void)
//
//  template <class Function >
//  uint32_t pop_back_if ( Function &&  M1)
//
//  uint32_t pop_copy_back ( value_type & V)
//
//  template <class Function >
//  uint32_t pop_copy_back_if ( value_type & V, Function && M1)
//
//  uint32_t pop_move_back ( value_type & V)
//
//  template <class Function >
//  uint32_t pop_move_back_if ( value_type & V, Function && M1)
//
//***************************************************************************
//
//---------------------------------------------------------------------------
//  function :pop_back
/// @brief erase the last element of the container
/// @param [in] none
/// @return none
/// @remarks This operation is O(constant)
//---------------------------------------------------------------------------
void pop_back ( void)
{
    std::lock_guard<spinlock_t>  S(spl);
    V.pop_back() ;
}
//
//---------------------------------------------------------------------------
//  function :pop_copy_back
/// @brief erase the last element of the tree and return a copy
/// @param [out] V : reference to a variable where copy the element
/// @return code of the operation
///         0- Element erased
///         1 - Empty tree
/// @remarks This operation is O(1)
//---------------------------------------------------------------------------
bool pop_copy_back ( value_type & P)
{
    //-------------------------- begin -----------------------------
    std::lock_guard<spinlock_t>  S(spl);
    if ( V.size() == 0) return false ;
    P = V.back() ;
    V.pop_back() ;
    return true;
}
//
//---------------------------------------------------------------------------
//  function :pop_copy_back
/// @brief erase the last NElem element of the stack, if possible, and push
///        back a copy in the vector V1
/// @param [in/out] V1 : vector where copy the elements extracted
/// @param [in] NElem : number of elements to extract, if possible
/// @return Number of elements extracted ( 0- Indicates that V is empty)
//---------------------------------------------------------------------------
template <class Allocator2>
size_type pop_copy_back ( std::vector<value_type,Allocator2> & V1,
                          size_type NElem )
{
    //-------------------------- begin -----------------------------
    std::lock_guard<spinlock_t>  S(spl);
    size_type Aux = 0 ;
    if ( V.size() != 0 )
    {   Aux = ( V.size() < NElem )? V.size() :NElem ;
        size_type PosIni = V.size() - Aux ;
        for ( size_type i = PosIni ; i < V.size() ; ++i)
            V1.push_back ( V [i]);
        V.erase( V.begin() + PosIni , V.end() );
    }
    return Aux;
}

}; // end class stack_cnc

//***************************************************************************
}// end namespace tools
}}// end HPX_INLINE_NAMESPACE(v2)
}// end namespace parallel
}// end namespace hpx
//***************************************************************************
#endif
