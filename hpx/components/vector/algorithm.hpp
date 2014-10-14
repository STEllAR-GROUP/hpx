//  Copyright (c) 2014 Anuj R. Sharma
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// This file contains the hpx implementation of some algorithm from Standard
// Template Library
//

#ifndef ALGORITHM_HPP
#define ALGORITHM_HPP

/** @file hpx/components/vector/algorithm.hpp
 *
 *  @brief This file contain the implementation of standard algorithm from stl
 *          library.
 *
 *  This file contain the implementation of the standard algorithm in stl in hpx
 *  and also contain their asynchronous API. It also contain some additional
 *  algorithm from thrust library.
 */

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>

#include <vector>

#include <hpx/components/vector/chunk_vector_component.hpp>
#include <hpx/components/vector/segmented_iterator.hpp>

#include <iostream>
//#define VALUE_TYPE double;

namespace hpx
{
    // PROGRAMMER DOCUMENTATION:
    // The idea of this implementation is taken from [Specially for for_each()]
    // http://lafstern.org/matt/segmented.pdf fill algorithm. [page no 7-8]
    //

    typedef typename boost::mpl::true_    true_type;
    typedef typename boost::mpl::false_   false_type;


    template <class iterator>
    struct segmented_iterator_traits{
        typedef false_type is_const_segmented_iterator;
    };

    template<>
    struct segmented_iterator_traits<hpx::const_segmented_vector_iterator>{
        typedef true_type is_const_segmented_iterator;
    };

    /** @brief Apply the function \a fn to each element in the
     *         range [first, last).
     *
     *  @tparam input_iterator  Segmented iterator to the sequence
     *
     *  @tparam fun             Unary Function returning void
     *
     *  @param first    Input iterator to the initial position of the in
     *                  the sequence
     *                  [Note the first position in the vector is 0]
     *  @param last     Input iterator to the final position of the in
     *                  the sequence [Note the last element is not inclusive
     *                  in the range[first, last)]
     *  @param fn       Unary function (either function pointer or move
     *                  constructible function object) that accept an
     *                  element in the range as argument.
     */
    template<class input_iterator, class fun>
    void for_eachT(  input_iterator first,
                    input_iterator last,
                    fun fn,
                    false_type)
    {
        auto sfirst_ = input_iterator::segment(first);
        auto slast_ = input_iterator::segment(last);
        
        std::vector<hpx::lcos::future<void>> for_each_lazy_sync;

        if(sfirst_ == slast_)
        {
            for_each_lazy_sync.push_back(
                hpx::stubs::chunk_vector::chunk_for_each_async(
                    (input_iterator::local(first).first).get(), //gives gid
                     input_iterator::local(first).second, // gives first index
                     input_iterator::local(last).second, //gives last index
                     fn) //gives the functor
                                         );
        }
        else
        {
            for_each_lazy_sync.push_back(
                hpx::stubs::chunk_vector::chunk_for_each_async(
                    (input_iterator::local(first).first).get(), //gives gid
                     input_iterator::local(first).second, // gives first index
                     input_iterator::end(sfirst_).second, //gives last index
                     fn) //gives the functor
                                         );
            ++sfirst_;
            while(sfirst_ != slast_)
            {
                for_each_lazy_sync.push_back(
                    hpx::stubs::chunk_vector::chunk_for_each_async(
                        (input_iterator::begin(sfirst_).first).get(), //gives gid
                         input_iterator::begin(sfirst_).second, // gives first index
                         input_iterator::end(sfirst_).second, //gives last index
                         fn) //gives the functor
                                              );
                ++sfirst_;
            }
            for_each_lazy_sync.push_back(
                hpx::stubs::chunk_vector::chunk_for_each_async(
                    (input_iterator::begin(slast_).first).get(), //gives gid
                     input_iterator::begin(slast_).second, // gives first index
                     input_iterator::local(last).second, //gives last index
                     fn) //gives the functor
                                         );
        }//end of else

        hpx::wait_all(for_each_lazy_sync);
    }//end of for_each


    /** @brief Apply the function \a fn to each element in the
     *         range [first, last).
     *
     *  @tparam const_input_iterator  Constant Segmented iterator to the sequence
     *
     *  @tparam fun             Unary Function returning void
     *
     *  @param first    Constant input iterator to the initial position of the in
     *                  the sequence
     *                  [Note the first position in the vector is 0]
     *  @param last     Constant input iterator to the final position of the in
     *                  the sequence [Note the last element is not inclusive
     *                  in the range[first, last)]
     *  @param fn       Unary function (either function pointer or move
     *                  constructible function object) that accept an
     *                  const element in the range as argument.
     */
    template<class const_input_iterator, class fun>
    void for_eachT(  const_input_iterator first,
                    const_input_iterator last,
                    fun fn,
                    true_type)
    {
        auto sfirst_ = const_input_iterator::segment(first);
        auto slast_ = const_input_iterator::segment(last);
        std::vector<hpx::lcos::future<void>> for_each_lazy_sync;

        if(sfirst_ == slast_)
        {
            for_each_lazy_sync.push_back(
                hpx::stubs::chunk_vector::chunk_for_each_const_async(
                    (const_input_iterator::local(first).first).get(), //gives gid
                     const_input_iterator::local(first).second, // gives first index
                     const_input_iterator::local(last).second, //gives last index
                     fn) //gives the functor
                                         );
        }
        else
        {
            for_each_lazy_sync.push_back(
                hpx::stubs::chunk_vector::chunk_for_each_const_async(
                    (const_input_iterator::local(first).first).get(), //gives gid
                     const_input_iterator::local(first).second, // gives first index
                     const_input_iterator::end(sfirst_).second, //gives last index
                     fn) //gives the functor
                                         );
            ++sfirst_;
            while(sfirst_ != slast_)
            {
                for_each_lazy_sync.push_back(
                    hpx::stubs::chunk_vector::chunk_for_each_const_async(
                        (const_input_iterator::begin(sfirst_).first).get(), //gives gid
                         const_input_iterator::begin(sfirst_).second, // gives first index
                         const_input_iterator::end(sfirst_).second, //gives last index
                         fn) //gives the functor
                                              );
                ++sfirst_;
            }
            for_each_lazy_sync.push_back(
                hpx::stubs::chunk_vector::chunk_for_each_const_async(
                    (const_input_iterator::begin(slast_).first).get(), //gives gid
                     const_input_iterator::begin(slast_).second, // gives first index
                     const_input_iterator::local(last).second, //gives last index
                     fn) //gives the functor
                                         );
        }//end of else

        hpx::wait_all(for_each_lazy_sync);
    }//end of for_each

    template <class iter, class fun>
    inline void for_each(iter first, iter last, fun fn)
    {
        typedef segmented_iterator_traits<iter> traits;
        for_eachT(first, last, fn, typename traits::is_const_segmented_iterator());
    }



    /** @brief Apply the function \a fn to each element in the
     *         range [first, last).
     *
     *  @tparam input_iterator  Segmented iterator to the sequence
     *
     *  @tparam fun             Unary Function returning void
     *
     *  @param first    Input iterator to the initial position of the in
     *                  the sequence
     *                  [Note the first position in the vector is 0]
     *  @param last     Input iterator to the final position of the in
     *                  the sequence [Note the last element is not inclusive
     *                  in the range[first, last)]
     *  @param fn       Unary function (either function pointer or move
     *                  constructible function object) that accept an
     *                  element in the range as argument.
     *  @return This return the hpx::future of type void
     *           [The void return type can help to check whether the action
     *           is completed or not]
     */
    template<class input_iterator, class fun>
    hpx::lcos::future<void>
     for_each_async(input_iterator first,
                    input_iterator last,
                    fun fn)
    {
        typedef segmented_iterator_traits<input_iterator> traits;
        typedef typename traits::is_const_segmented_iterator iter_type;
        return hpx::async(launch::async,
                          hpx::util::bind((&hpx::for_each<input_iterator, fun>),
                                           first,
                                           last,
                                           fn
                                          )
                          );
    }//end of for_each_async

    /** @brief Apply the function \a fn to each element in the range
     *          [first, first + n).
     *
     *  @tparam input_iterator  Segmented iterator to the sequence
     *
     *  @tparam fun             Unary Function returning void
     *
     *  @param first    Input iterator to the initial position of the in
     *                   the sequence [Note the first position in the vector
     *                   is 0]
     *  @param n        The size of input sequence
     *  @param fn       Unary function (either function pointer or move
     *                   constructible function object) that accept an
     *                   element in the range as argument.
     */
    template<class input_iterator, class fun>
    void for_each_n(input_iterator first,
                    std::size_t n,
                    fun fn)
    {
        hpx::for_each(first, first + n, fn);
    }//end of for_each_n

    /** @brief Apply the function \a fn to each element in the range
     *          [first, first + n).
     *
     *  @tparam input_iterator  Segmented iterator to the sequence
     *
     *  @tparam fun             Unary Function returning void
     *
     *  @param first    Input iterator to the initial position of the in
     *                   the sequence [Note the first position in the vector
     *                   is 0]
     *  @param n        The size of input sequence
     *  @param fn       Unary function (either function pointer or move
     *                   constructible function object) that accept an
     *                   element in the range as argument.
     *
     *  @return This return the hpx::future of type void [The void return
     *           type can help to check whether the action is completed or
     *           not]
     */
    template<class input_iterator, class fun>
    hpx::lcos::future<void>
     for_each_n_async( input_iterator first,
                       std::size_t n,
                       fun fn)
    {
        typedef segmented_iterator_traits<input_iterator> traits;
        typedef typename traits::is_const_segmented_iterator iter_type;
        return hpx::async(launch::async,
                          hpx::util::bind((&hpx::for_each<input_iterator, fun>),
                                           first,
                                           first + n,
                                           fn
                                          )
                          );
    }//end of for_each_n_async

}//end of hpx namespace

#endif // ALGORITHM_HPP
