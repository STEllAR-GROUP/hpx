//  Copyright (c) 2015-2017 Francisco Jose Tapia
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/parallel/algorithms/detail/insertion_sort.hpp>
#include <hpx/parallel/algorithms/detail/is_sorted.hpp>
#include <hpx/parallel/util/nbits.hpp>
#include <hpx/parallel/util/range.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::parallel::detail {

    /// This function divide r_input in two parts, sort it, and merge moving
    ///        the elements to range_buf
    /// \param [in] rng_a       range with the elements to sort
    /// \param [in] rng_b       range with the elements sorted
    /// \param [in] comp        object for to compare two elements
    /// \param [in] level       when is 0, sort with the insertion_sort
    ///                         algorithm if not make a recursive call swapping
    ///                         the ranges
    /// \return range with all the elements sorted and moved
    template <typename Iter1, typename Sent1, typename Iter2, typename Sent2,
        typename Compare>
    void range_sort(util::range<Iter1, Sent1> const& rng_a,
        util::range<Iter2, Sent2> const& rng_b, Compare comp,
        std::uint32_t level)
    {
        using range_it1 = util::range<Iter1, Sent1>;
        using range_it2 = util::range<Iter2, Sent2>;

        std::size_t nelem1 = (rng_a.size() + 1) >> 1;

        range_it1 rng_a1(rng_a.begin(), rng_a.begin() + nelem1);
        range_it1 rng_a2(rng_a.begin() + nelem1, rng_a.end());

        if (level < 2)
        {
            insertion_sort(rng_a1.begin(), rng_a1.end(), comp);
            insertion_sort(rng_a2.begin(), rng_a2.end(), comp);
        }
        else
        {
            range_sort(range_it2(rng_b.begin(), rng_b.begin() + nelem1), rng_a1,
                comp, level - 1);
            range_sort(range_it2(rng_b.begin() + nelem1, rng_b.end()), rng_a2,
                comp, level - 1);
        }

        parallel::util::full_merge(rng_b, rng_a1, rng_a2, comp);
    }

    /// \struct spin_sort_helper
    /// \brief this is a struct for to do a stable sort exception safe
    /// \tparam Iter : iterator to the elements
    /// \tparam Compare : object for to Compare the elements pointed by Iter
    /// \remarks
    template <typename Iter, typename Sent, typename Compare>
    class spin_sort_helper
    {
        using value_type = typename std::iterator_traits<Iter>::value_type;
        using range_it = util::range<Iter, Sent>;
        using range_buf = util::range<value_type*>;

        static constexpr std::uint32_t sort_min = 36;

        value_type* ptr;
        std::size_t nptr;
        bool construct = false;
        bool owner = false;

        /// \brief constructor of the struct
        /// \param [in] first : range of elements to sort
        /// \param [in] last : range of elements to sort
        /// \param [in] comp : object for to compare two elements
        /// \param [in] paux
        /// \param [in] naux
        spin_sort_helper(Iter first, Sent last, Compare comp, value_type* paux,
            std::size_t naux);

    public:
        /// \brief constructor of the struct
        /// \param [in] first : range of elements to sort
        /// \param [in] last : range of elements to sort
        /// \param [in] comp : object for to Compare two elements
        spin_sort_helper(Iter first, Sent last, Compare comp = Compare())
          : spin_sort_helper(first, last, comp, nullptr, 0)
        {
        }

        /// \brief constructor of the struct
        /// \param [in] first : range of elements to sort
        /// \param [in] last : range of elements to sort
        /// \param [in] comp : object for to Compare two elements
        /// \param [in] range_aux : range used as auxiliary memory
        spin_sort_helper(
            Iter first, Sent last, Compare comp, range_buf range_aux)
          : spin_sort_helper(first, last, comp, range_aux.begin(),
                static_cast<std::size_t>(range_aux.size()))
        {
        }

        /// \brief destructor of the struct. Deallocate all the data structure
        ///        used in the sorting
        /// \exception
        /// \return
        /// \remarks
        ~spin_sort_helper()
        {
            if (construct)
            {
                parallel::util::destroy_range(
                    util::range<value_type*>(ptr, ptr + nptr));
                construct = false;
            }

            if (owner)
            {
                std::free(ptr);
            }
        }
    };    // End of class spin_sort_helper

    /// \brief constructor of the struct
    /// \param [in] first : range of elements to sort
    /// \param [in] last : range of elements to sort
    /// \param [in] comp : object for to Compare two elements
    /// \param [in] paux
    /// \param [in] naux
    template <typename Iter, typename Sent, typename Compare>
    spin_sort_helper<Iter, Sent, Compare>::spin_sort_helper(
        Iter first, Sent last, Compare comp, value_type* paux, std::size_t naux)
      : ptr(paux)
      , nptr(naux)
    {
        util::range<Iter> r_input(first, last);
        HPX_ASSERT(r_input.size() >= 0);

        std::size_t const nelem = r_input.size();
        owner = construct = false;

        nptr = (nelem + 1) >> 1;
        std::size_t nelem_1 = nptr;
        std::size_t nelem_2 = nelem - nelem_1;

        if (nelem <= (sort_min << 1))
        {
            insertion_sort(first, last, comp);
            return;
        }

        if (detail::is_sorted_sequential(first, last, comp))
            return;

        if (ptr == nullptr)
        {
            // acquire uninitialized memory
            ptr = static_cast<value_type*>(
                std::malloc(nptr * sizeof(value_type)));
            if (ptr == nullptr)
            {
                throw std::bad_alloc();
            }
            owner = true;
        }

        range_buf rng_buf(ptr, ptr + nptr);

        std::uint32_t nlevel =
            util::nbits64((nelem + sort_min - 1) / sort_min - 1) - 1;
        HPX_ASSERT(nlevel != 0);

        if ((nlevel & 1) == 1)
        {
            // if the number of levels is odd, the data are in the first parameter
            // of range_sort, and the results appear in the second parameter
            range_it rng_a1(first, first + nelem_2);
            range_it rng_a2(first + nelem_2, last);

            rng_buf = parallel::util::uninit_move(rng_buf, rng_a2);
            construct = true;

            range_sort(rng_buf, rng_a2, comp, nlevel);
            range_buf rng_bx(rng_buf.begin(), rng_buf.begin() + nelem_2);

            range_sort(rng_a1, rng_bx, comp, nlevel);
            parallel::util::half_merge(r_input, rng_bx, rng_a2, comp);
        }
        else
        {
            // If the number of levels is even, the data are in the second
            // parameter of range_sort, and the results are in the same parameter
            range_it rng_a1(first, first + nelem_1);
            range_it rng_a2(first + nelem_1, last);

            rng_buf = parallel::util::uninit_move(rng_buf, rng_a1);
            construct = true;

            range_sort(rng_a1, rng_buf, comp, nlevel);

            rng_a1 = range_it(rng_a1.begin(), rng_a1.begin() + rng_a2.size());
            range_sort(rng_a1, rng_a2, comp, nlevel);
            parallel::util::half_merge(r_input, rng_buf, rng_a2, comp);
        }
    }

    template <typename Iter, typename Sent>
    void spin_sort(Iter first, Sent last)
    {
        using compare =
            std::less<typename std::iterator_traits<Iter>::value_type>;

        spin_sort_helper<Iter, Sent, compare> sorter(first, last, compare{});
    }

    template <typename Iter, typename Sent, typename Compare>
    void spin_sort(Iter first, Sent last, Compare&& comp)
    {
        spin_sort_helper<Iter, Sent, std::decay_t<Compare>> sorter(
            first, last, HPX_FORWARD(Compare, comp));
    }

    template <typename Iter, typename Sent, typename Compare>
    void spin_sort(Iter first, Sent last, Compare&& comp,
        util::range<typename std::iterator_traits<Iter>::value_type*> range_aux)
    {
        spin_sort_helper<Iter, Sent, std::decay_t<Compare>> sorter(
            first, last, HPX_FORWARD(Compare, comp), range_aux);
    }

    template <typename Iter, typename Sent, typename Compare>
    void spin_sort(Iter first, Sent last, Compare comp,
        typename std::iterator_traits<Iter>::value_type* paux, std::size_t naux)
    {
        spin_sort_helper<Iter, Sent, std::decay_t<Compare>> sorter(
            first, last, HPX_FORWARD(Compare, comp), paux, naux);
    }

}    // namespace hpx::parallel::detail
