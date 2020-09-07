//  Copyright (c) 2015-2017 Francisco Jose Tapia
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace util {

    /// \brief create an object in the memory specified by ptr
    /// \tparam Value : typename of the object to create
    /// \tparam Args : parameters for the constructor
    /// \param [in] ptr : pointer to the memory where to create the object
    /// \param [in] args : arguments to the constructor
    template <typename Value, typename... Args>
    inline void construct_object(Value* ptr, Args&&... args)
    {
        (::new (static_cast<void*>(ptr)) Value(std::forward<Args>(args)...));
    }

    /// \brief destroy an object in the memory specified by ptr
    /// \tparam Value : typename of the object to create
    /// \param [in] ptr : pointer to the object to destroy
    //-----------------------------------------------------------------------------
    template <typename Value>
    inline void destroy_object(Value* ptr)
    {
        ptr->~Value();
    }

    /// Initialize a range of objects with the object val moving across them
    /// \param [in] r : range of elements not initialized
    /// \param [in] val : object used for the initialization
    /// \return range initialized
    template <typename Iter, typename Sent>
    inline void init(Iter first, Sent last,
        typename std::iterator_traits<Iter>::value_type& val)
    {
        if (first == last)
        {
            return;
        }

        construct_object(&(*first), std::move(val));

        Iter it1 = first, it2 = first + 1;
        while (it2 != last)
        {
            construct_object(&(*(it2++)), std::move(*(it1++)));
        }

        val = std::move(*(last - 1));
    }

    /// \brief create an object in the memory specified by ptr
    /// \tparam Value : typename of the object to create
    /// \tparam Args : parameters for the constructor
    /// \param [in] ptr : pointer to the memory where to create the object
    /// \param [in] args : arguments to the constructor
    template <typename Value, typename... Args>
    inline void construct(Value* ptr, Args&&... args)
    {
        (::new (static_cast<void*>(ptr)) Value(std::forward<Args>(args)...));
    }

    /// \brief Move objects
    /// \tparam Iter : iterator to the elements
    /// \tparam Value : typename of the object to create
    /// \param [in] itdest : iterator to the final place of the objects
    /// \param [in] R : range to move
    template <typename Iter1, typename Sent1, typename Iter2>
    inline Iter2 init_move(Iter2 it_dest, Iter1 first, Sent1 last)
    {
        while (first != last)
        {
            *(it_dest++) = std::move(*(first++));
        }
        return it_dest;
    }

    /// \brief Move objects to uninitialized memory
    /// \tparam Iter : iterator to the elements
    /// \tparam Value : typename of the object to construct
    /// \param [in] ptr : pointer to the memory where to create the object
    /// \param [in] R : range to move
    template <typename Iter, typename Sent,
        typename Value = typename std::iterator_traits<Iter>::value_type>
    inline Value* uninit_move(Value* ptr, Iter first, Sent last)
    {
        using value_type = typename std::iterator_traits<Iter>::value_type;

        static_assert(
            std::is_same<Value, value_type>::value, "Incompatible iterators\n");

        while (first != last)
        {
            ::new (static_cast<void*>(ptr++)) Value(std::move(*(first++)));
        }

        return ptr;
    }

    /// \brief Move objects to uninitialized memory
    /// \tparam Iter : iterator to the elements
    /// \tparam Value : typename of the object to construct
    /// \param [in] ptr : pointer to the memory where to construct the object
    /// \param [in] R : range to move
    template <typename Iter, typename Sent>
    inline void destroy(Iter first, Sent last)
    {
        using value_type = typename std::iterator_traits<Iter>::value_type;
        while (first != last)
        {
            (&(*(first++)))->~value_type();
        }
    }

    /// \brief Merge two contiguous buffers pointed by buf1 and buf2 , and put
    ///        in the buffer pointed by buf_out
    /// \param [in] buf1 : iterator to the first element in the first buffer
    /// \param [in] end_buf1 : final iterator of first buffer
    /// \param [in] buf2 : iterator to the first iterator to the second buffer
    /// \param [in] end_buf2 : final iterator of the second buffer
    /// \param [in] buf_out : buffer where move the elements merged
    /// \param [in] comp : comparison object
    template <typename Iter1, typename Sent1, typename Iter2, typename Compare>
    inline Iter2 full_merge(Iter1 buf1, Sent1 end_buf1, Iter1 buf2,
        Sent1 end_buf2, Iter2 buf_out, Compare comp)
    {
        using value1_t = typename std::iterator_traits<Iter1>::value_type;
        using value2_t = typename std::iterator_traits<Iter2>::value_type;

        static_assert(std::is_same<value1_t, value2_t>::value,
            "Incompatible iterators\n");

        while ((buf1 != end_buf1) && (buf2 != end_buf2))
        {
            *(buf_out++) = (!comp(*buf2, *buf1)) ? std::move(*(buf1++)) :
                                                   std::move(*(buf2++));
        }
        return (buf1 == end_buf1) ? init_move(buf_out, buf2, end_buf2) :
                                    init_move(buf_out, buf1, end_buf1);
    }

    /// \brief Merge two contiguous buffers pointed by first1 and first2 , and put
    ///        in the uninitialized buffer pointed by it_out
    /// \param [in] first1 : iterator to the first element in the first buffer
    /// \param [in] last : last iterator of the first buffer
    /// \param [in] first2 : iterator to the first element to the second buffer
    /// \param [in] last22 : final iterator of the second buffer
    /// \param [in] it_out : uninitialized buffer where move the elements merged
    /// \param [in] comp : comparison object
    template <typename Iter, typename Sent, typename Value, typename Compare>
    inline Value* uninit_full_merge(Iter first1, Sent last1, Iter first2,
        Sent last2, Value* it_out, Compare comp)
    {
        using type1 = typename std::iterator_traits<Iter>::value_type;

        static_assert(
            std::is_same<Value, type1>::value, "Incompatible iterators\n");

        while (first1 != last1 && first2 != last2)
        {
            construct((it_out++),
                (!comp(*first2, *first1)) ? std::move(*(first1++)) :
                                            std::move(*(first2++)));
        };
        return (first1 == last1) ? uninit_move(it_out, first2, last2) :
                                   uninit_move(it_out, first1, last1);
    }

    /// \brief : Merge two buffers. The first buffer is in a separate memory.
    ///          The second buffer have a empty space before buf2 of the same size
    ///          than the (end_buf1 - buf1)
    /// \param [in] buf1 : iterator to the first element of the first buffer
    /// \param [in] end_buf1 : iterator to the last element of the first buffer
    /// \param [in] buf2 : iterator to the first element of the second buffer
    /// \param [in] end_buf2 : iterator to the last element of the second buffer
    /// \param [in] buf_out : iterator to the first element to the buffer where put
    ///                       the result
    /// \param [in] comp : object for Compare two elements of the type pointed
    ///                    by the Iter1 and Iter2
    /// \remarks The elements pointed by Iter1 and Iter2 must be the same
    template <typename Iter1, typename Sent1, typename Iter2, typename Sent2,
        typename Compare>
    inline Iter2 half_merge(Iter1 buf1, Sent1 end_buf1, Iter2 buf2,
        Sent2 end_buf2, Iter2 buf_out, Compare comp)
    {
        using value1_t = typename std::iterator_traits<Iter1>::value_type;
        using value2_t = typename std::iterator_traits<Iter2>::value_type;

        static_assert(std::is_same<value1_t, value2_t>::value,
            "Incompatible iterators\n");

        while ((buf1 != end_buf1) && (buf2 != end_buf2))
        {
            *(buf_out++) = (!comp(*buf2, *buf1)) ? std::move(*(buf1++)) :
                                                   std::move(*(buf2++));
        }
        return (buf2 == end_buf2) ? init_move(buf_out, buf1, end_buf1) :
                                    end_buf2;
    }

    /// Merge two non contiguous buffers, placing the results in the buffers
    ///          for to do this use an auxiliary buffer pointed by aux
    /// \param [in] src1 : iterator to the first element of the first buffer
    /// \param [in] end_src1 : last iterator  of the first buffer
    /// \param [in] src2 : iterator to the first element of the second buffer
    /// \param [in] end_src2 : last iterator  of the second buffer
    /// \param [in] aux  : iterator to the first element of the auxiliary buffer
    /// \param [in] comp : object for to Compare elements
    /// \exception
    /// \return true : not changes done
    ///         false : changes in the buffers
    /// \remarks
    template <typename Iter1, typename Sent1, typename Iter2, typename Sent2,
        typename Iter3, typename Compare>
    bool in_place_merge_uncontiguous(Iter1 src1, Sent1 end_src1, Iter2 src2,
        Sent2 end_src2, Iter3 aux, Compare comp)
    {
        using type1 = typename std::iterator_traits<Iter1>::value_type;
        using type2 = typename std::iterator_traits<Iter2>::value_type;
        using type3 = typename std::iterator_traits<Iter3>::value_type;

        static_assert(
            std::is_same<type1, type2>::value, "Incompatible iterators\n");
        static_assert(
            std::is_same<type3, type2>::value, "Incompatible iterators\n");

        if (src1 == end_src1 || src2 == end_src2 ||
            !comp(*src2, *(end_src1 - 1)))
        {
            return true;
        }

        while (src1 != end_src1 && !comp(*src2, *src1))
        {
            ++src1;
        }

        Iter3 const end_aux = aux + (end_src1 - src1);
        Iter2 src2_first = src2;
        init_move(aux, src1, end_src1);

        while ((src1 != end_src1) && (src2 != end_src2))
        {
            *(src1++) = std::move((!comp(*src2, *aux)) ? *(aux++) : *(src2++));
        }

        if (src2 == end_src2)
        {
            while (src1 != end_src1)
            {
                *(src1++) = std::move(*(aux++));
            }
            init_move(src2_first, aux, end_aux);
        }
        else
        {
            half_merge(aux, end_aux, src2, end_src2, src2_first, comp);
        }
        return false;
    }

    /// \brief : merge two contiguous buffers,using an auxiliary buffer pointed
    ///          by buf
    ///
    /// \param [in] src1: iterator to the first position of the first buffer
    /// \param [in] src2: final iterator of the first buffer and first iterator
    ///                   of the second buffer
    /// \param [in] end_src2 : final iterator of the second buffer
    /// \param [in] buf  : iterator to buffer used as auxiliary memory
    /// \param [in] comp : object for to Compare elements
    /// \exception
    /// \return true : not changes done
    ///         false : changes in the buffers
    /// \remarks
    template <typename Iter1, typename Sent1, typename Iter2, typename Compare>
    inline bool in_place_merge(
        Iter1 src1, Iter1 src2, Sent1 end_src2, Iter2 buf, Compare comp)
    {
        using type1 = typename std::iterator_traits<Iter1>::value_type;
        using type2 = typename std::iterator_traits<Iter2>::value_type;

        static_assert(
            std::is_same<type1, type2>::value, "Incompatible iterators\n");

        if (src1 == src2 || src2 == end_src2 || !comp(*src2, *(src2 - 1)))
        {
            return true;
        }

        Iter1 end_src1 = src2;
        while (src1 != end_src1 && !comp(*src2, *src1))
        {
            ++src1;
        }

        if (src1 == end_src1)
        {
            return false;
        }

        size_t nx = end_src1 - src1;
        init_move(buf, src1, end_src1);
        half_merge(buf, buf + nx, src2, end_src2, src1, comp);
        return false;
    }
}}}    // namespace hpx::parallel::util
