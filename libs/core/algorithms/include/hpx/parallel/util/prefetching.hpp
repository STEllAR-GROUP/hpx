//  Copyright (c) 2016 Zahra Khatami, Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/type_support/pack.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(HPX_HAVE_MM_PREFETCH)
#if defined(HPX_MSVC)
#include <intrin.h>
#endif
#if defined(HPX_GCC_VERSION)
#include <emmintrin.h>
#endif
#endif

namespace hpx { namespace parallel { namespace util {

    namespace prefetching {

        template <typename Itr, typename... Ts>
        class prefetching_iterator
        {
        public:
            using base_iterator = Itr;

            using iterator_category = std::random_access_iterator_tag;
            using value_type = typename std::iterator_traits<Itr>::value_type;
            using difference_type = std::ptrdiff_t;
            using pointer = value_type*;
            using reference = value_type&;

        private:
            using ranges_type = hpx::tuple<std::reference_wrapper<Ts>...>;

            ranges_type rngs_;
            base_iterator base_;
            std::size_t chunk_size_;
            std::size_t range_size_;
            std::size_t idx_;

        public:
            // different versions of clang-format do different things
            // clang-format off
            explicit prefetching_iterator(std::size_t idx, base_iterator base,
                std::size_t chunk_size, std::size_t range_size,
                ranges_type const& rngs)
              : rngs_(rngs)
              , base_(base)
              , chunk_size_(chunk_size)
              , range_size_(range_size)
              , idx_((std::min) (idx, range_size))
            {
            }
            // clang-format on

            ranges_type const& ranges() const
            {
                return rngs_;
            }

            Itr base() const
            {
                return base_;
            }
            std::size_t chunk_size() const
            {
                return chunk_size_;
            }
            std::size_t range_size() const
            {
                return range_size_;
            }
            std::size_t index() const
            {
                return idx_;
            }

            inline prefetching_iterator& operator+=(difference_type rhs)
            {
                // different versions of clang-format do different things
                // clang-format off
                std::size_t last =
                    (std::min) (idx_ + rhs * chunk_size_, range_size_);
                // clang-format on

                std::advance(base_, last - idx_);
                idx_ = last;
                return *this;
            }
            inline prefetching_iterator& operator-=(difference_type rhs)
            {
                std::size_t first = 0;
                if (idx_ > rhs * chunk_size_)
                    first = idx_ - rhs * chunk_size_;

                std::advance(base_, idx_ - first);
                idx_ = first;
                return *this;
            }
            inline prefetching_iterator& operator++()
            {
                *this += 1;
                return *this;
            }
            inline prefetching_iterator& operator--()
            {
                *this -= 1;
                return *this;
            }
            inline prefetching_iterator operator++(int)
            {
                prefetching_iterator tmp(*this);
                operator++();
                return tmp;
            }
            inline prefetching_iterator operator--(int)
            {
                prefetching_iterator tmp(*this);
                operator--();
                return tmp;
            }

            inline difference_type operator-(
                const prefetching_iterator& rhs) const
            {
                // round up distance to cover all of underlying range
                return (idx_ - rhs.idx_ + chunk_size_ - 1) / chunk_size_;
            }

            friend inline prefetching_iterator operator+(
                prefetching_iterator const& lhs, difference_type rhs)
            {
                prefetching_iterator tmp(lhs);
                tmp += rhs;
                return tmp;
            }
            friend inline prefetching_iterator operator-(
                prefetching_iterator const& lhs, difference_type rhs)
            {
                prefetching_iterator tmp(lhs);
                tmp -= rhs;
                return tmp;
            }

            // FIXME: should other members be compared too?
            inline bool operator==(const prefetching_iterator& rhs) const
            {
                return idx_ == rhs.idx_ && base_ == rhs.base_;
            }

            // FIXME: should the base iterators be compared too?
            inline bool operator!=(const prefetching_iterator& rhs) const
            {
                return idx_ != rhs.idx_;
            }
            inline bool operator>(const prefetching_iterator& rhs) const
            {
                return idx_ > rhs.idx_;
            }
            inline bool operator<(const prefetching_iterator& rhs) const
            {
                return idx_ < rhs.idx_;
            }
            inline bool operator>=(const prefetching_iterator& rhs) const
            {
                return idx_ >= rhs.idx_;
            }
            inline bool operator<=(const prefetching_iterator& rhs) const
            {
                return idx_ <= rhs.idx_;
            }

            // FIXME: This looks wrong, it should dispatch to the base iterator
            //        instead.
            inline std::size_t& operator[](std::size_t)
            {
                return idx_;
            }

            // FIXME: This looks wrong, it should dispatch to the base iterator
            //        instead.
            inline std::size_t operator*() const
            {
                return idx_;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // Helper class to initialize prefetching_iterator
        template <typename Itr, typename... Ts>
        struct prefetcher_context
        {
        private:
            typedef hpx::tuple<std::reference_wrapper<Ts>...> ranges_type;

            Itr it_begin_;
            Itr it_end_;
            ranges_type rngs_;
            std::size_t chunk_size_;
            std::size_t range_size_;

            static constexpr std::size_t sizeof_first_value_type =
                sizeof(typename hpx::tuple_element<0, ranges_type>::type::type);

        public:
            prefetcher_context(Itr begin, Itr end, ranges_type const& rngs,
                std::size_t p_factor = 1)
              : it_begin_(begin)
              , it_end_(end)
              , rngs_(rngs)
              , chunk_size_((p_factor * threads::get_cache_line_size()) /
                    sizeof_first_value_type)
              , range_size_(std::distance(begin, end))
            {
            }

            prefetching_iterator<Itr, Ts...> begin()
            {
                return prefetching_iterator<Itr, Ts...>(
                    0ull, it_begin_, chunk_size_, range_size_, rngs_);
            }

            prefetching_iterator<Itr, Ts...> end()
            {
                return prefetching_iterator<Itr, Ts...>(
                    range_size_, it_end_, chunk_size_, range_size_, rngs_);
            }
        };

        ///////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_MM_PREFETCH)
        template <typename... T>
        HPX_FORCEINLINE void prefetch_addresses(T const&... ts)
        {
            int const sequencer[] = {
                (_mm_prefetch(
                     const_cast<char*>((char const*) &ts), _MM_HINT_T0),
                    0)...,
                0};
            (void) sequencer;
        }

        template <typename... Ts, std::size_t... Is>
        HPX_FORCEINLINE void prefetch_containers(hpx::tuple<Ts...> const& t,
            hpx::util::index_pack<Is...>, std::size_t idx)
        {
            prefetch_addresses((hpx::get<Is>(t).get())[idx]...);
        }
#else
        template <typename... Ts, std::size_t... Is>
        HPX_FORCEINLINE void prefetch_containers(hpx::tuple<Ts...> const& t,
            hpx::util::index_pack<Is...>, std::size_t idx)
        {
            int const sequencer[] = {(hpx::get<Is>(t).get()[idx], 0)..., 0};
            (void) sequencer;
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        struct loop_n_helper
        {
            template <typename Itr, typename... Ts, typename F, typename Pred>
            static constexpr prefetching_iterator<Itr, Ts...> call(
                prefetching_iterator<Itr, Ts...> it, std::size_t count, F&& f,
                Pred)
            {
                using index_pack_type =
                    typename hpx::util::make_index_pack<sizeof...(Ts)>::type;

                for (/**/; count != 0; (void) --count, ++it)
                {
                    Itr base = it.base();
                    std::size_t j = it.index();

                    // different versions of clang-format do different things
                    // clang-format off
                    std::size_t last = (std::min) (it.index() + it.chunk_size(),
                        it.range_size());
                    // clang-format on

                    for (/**/; j != last; (void) ++j, ++base)
                    {
                        f(base);
                    }

                    if (j != it.range_size())
                    {
                        prefetch_containers(it.ranges(), index_pack_type(), j);
                    }
                }
                return it;
            }

            template <typename Itr, typename... Ts, typename CancelToken,
                typename F, typename Pred>
            static constexpr prefetching_iterator<Itr, Ts...> call(
                prefetching_iterator<Itr, Ts...> it, std::size_t count,
                CancelToken& tok, F&& f, Pred)
            {
                using index_pack_type =
                    typename hpx::util::make_index_pack<sizeof...(Ts)>::type;

                for (/**/; count != 0; (void) --count, ++it)
                {
                    if (tok.was_cancelled())
                        break;

                    Itr base = it.base();
                    std::size_t j = it.index();

                    // different versions of clang-format do different things
                    // clang-format off
                    std::size_t last = (std::min) (it.index() + it.chunk_size(),
                        it.range_size());
                    // clang-format on

                    for (/**/; j != last; (void) ++j, ++base)
                    {
                        f(base);
                    }

                    if (j != it.range_size())
                    {
                        prefetch_containers(it.ranges(), index_pack_type(), j);
                    }
                }
                return it;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Itr, typename... Ts, typename F>
        HPX_HOST_DEVICE
            HPX_FORCEINLINE constexpr prefetching_iterator<Itr, Ts...>
            tag_invoke(hpx::parallel::util::loop_n_t<ExPolicy>,
                prefetching_iterator<Itr, Ts...> it, std::size_t count, F&& f)
        {
            return loop_n_helper::call(
                it, count, HPX_FORWARD(F, f), std::true_type());
        }

        ///////////////////////////////////////////////////////////////////////
        struct loop_n_ind_helper
        {
            template <typename Itr, typename... Ts, typename F, typename Pred>
            static constexpr prefetching_iterator<Itr, Ts...> call(
                prefetching_iterator<Itr, Ts...> it, std::size_t count, F&& f,
                Pred)
            {
                using index_pack_type =
                    typename hpx::util::make_index_pack<sizeof...(Ts)>::type;

                for (/**/; count != 0; (void) --count, ++it)
                {
                    Itr base = it.base();
                    std::size_t j = it.index();

                    // different versions of clang-format do different things
                    // clang-format off
                    std::size_t last = (std::min) (it.index() + it.chunk_size(),
                        it.range_size());
                    // clang-format on

                    for (/**/; j != last; (void) ++j, ++base)
                    {
                        f(*base);
                    }

                    if (j != it.range_size())
                    {
                        prefetch_containers(it.ranges(), index_pack_type(), j);
                    }
                }
                return it;
            }

            template <typename Itr, typename... Ts, typename CancelToken,
                typename F, typename Pred>
            static constexpr prefetching_iterator<Itr, Ts...> call(
                prefetching_iterator<Itr, Ts...> it, std::size_t count,
                CancelToken& tok, F&& f, Pred)
            {
                using index_pack_type =
                    typename hpx::util::make_index_pack<sizeof...(Ts)>::type;

                for (/**/; count != 0; (void) --count, ++it)
                {
                    if (tok.was_cancelled())
                        break;

                    Itr base = it.base();
                    std::size_t j = it.index();

                    // different versions of clang-format do different things
                    // clang-format off
                    std::size_t last = (std::min) (it.index() + it.chunk_size(),
                        it.range_size());
                    // clang-format on

                    for (/**/; j != last; (void) ++j, ++base)
                    {
                        f(*base);
                    }

                    if (j != it.range_size())
                    {
                        prefetch_containers(it.ranges(), index_pack_type(), j);
                    }
                }
                return it;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Itr, typename... Ts, typename F>
        HPX_HOST_DEVICE
            HPX_FORCEINLINE constexpr prefetching_iterator<Itr, Ts...>
            tag_invoke(hpx::parallel::util::loop_n_ind_t<ExPolicy>,
                prefetching_iterator<Itr, Ts...> it, std::size_t count, F&& f)
        {
            return loop_n_ind_helper::call(
                it, count, HPX_FORWARD(F, f), std::true_type());
        }
    }    // namespace prefetching

    ///////////////////////////////////////////////////////////////////////////
    // function to create a prefetcher_context
    template <typename Itr, typename... Ts>
    prefetching::prefetcher_context<Itr, Ts const...> make_prefetcher_context(
        Itr base_begin, Itr base_end, std::size_t p_factor, Ts const&... rngs)
    {
        static_assert(hpx::traits::is_random_access_iterator<Itr>::value,
            "Iterators have to be of random access iterator category");
        static_assert(hpx::util::all_of<hpx::traits::is_range<Ts>...>::value,
            "All variadic parameters have to represent ranges");

        using ranges_type = hpx::tuple<std::reference_wrapper<Ts const>...>;

        auto&& ranges = ranges_type(std::cref(rngs)...);
        return prefetching::prefetcher_context<Itr, Ts const...>(
            base_begin, base_end, HPX_MOVE(ranges), p_factor);
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Itr, typename... Ts>
        struct loop<prefetching::prefetching_iterator<Itr, Ts...>>
        {
            typedef prefetching::prefetching_iterator<Itr, Ts...> iterator_type;
            typedef typename iterator_type::base_iterator type;
            typedef typename hpx::util::make_index_pack<sizeof...(Ts)>::type
                index_pack_type;

            template <typename End, typename F>
            static iterator_type call(iterator_type it, End end, F&& f)
            {
                for (/**/; it != end; ++it)
                {
                    Itr base = it.base();
                    std::size_t j = it.index();

                    // different versions of clang-format do different things
                    // clang-format off
                    std::size_t last = (std::min) (it.index() + it.chunk_size(),
                        it.range_size());
                    // clang-format on

                    for (/**/; j != last; (void) ++j, ++base)
                        f(base);

                    if (j != it.range_size())
                        prefetch_containers(it.ranges(), index_pack_type(), j);
                }
                return it;
            }

            template <typename End, typename CancelToken, typename F>
            static iterator_type call(
                iterator_type it, End end, CancelToken& tok, F&& f)
            {
                for (/**/; it != end; ++it)
                {
                    if (tok.was_cancelled())
                        break;

                    Itr base = it.base();
                    std::size_t j = it.index();

                    // different versions of clang-format do different things
                    // clang-format off
                    std::size_t last = (std::min) (it.index() + it.chunk_size(),
                        it.range_size());
                    // clang-format on

                    for (/**/; j != last; (void) ++j, ++base)
                        f(base);

                    if (j != it.range_size())
                        prefetch_containers(it.ranges(), index_pack_type(), j);
                }
                return it;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Itr, typename... Ts>
        struct loop_ind<prefetching::prefetching_iterator<Itr, Ts...>>
        {
            typedef prefetching::prefetching_iterator<Itr, Ts...> iterator_type;
            typedef typename iterator_type::base_iterator type;
            typedef typename hpx::util::make_index_pack<sizeof...(Ts)>::type
                index_pack_type;

            template <typename End, typename F>
            static iterator_type call(iterator_type it, End end, F&& f)
            {
                for (/**/; it != end; ++it)
                {
                    Itr base = it.base();
                    std::size_t j = it.index();

                    // different versions of clang-format do different things
                    // clang-format off
                    std::size_t last = (std::min) (it.index() + it.chunk_size(),
                        it.range_size());
                    // clang-format on

                    for (/**/; j != last; (void) ++j, ++base)
                        f(*base);

                    if (j != it.range_size())
                        prefetch_containers(it.ranges(), index_pack_type(), j);
                }
                return it;
            }

            template <typename End, typename CancelToken, typename F>
            static iterator_type call(
                iterator_type it, End end, CancelToken& tok, F&& f)
            {
                for (/**/; it != end; ++it)
                {
                    if (tok.was_cancelled())
                        break;

                    Itr base = it.base();
                    std::size_t j = it.index();

                    // different versions of clang-format do different things
                    // clang-format off
                    std::size_t last = (std::min) (it.index() + it.chunk_size(),
                        it.range_size());
                    // clang-format on

                    for (/**/; j != last; (void) ++j, ++base)
                        f(*base);

                    if (j != it.range_size())
                        prefetch_containers(it.ranges(), index_pack_type(), j);
                }
                return it;
            }
        };
    }    // namespace detail
}}}      // namespace hpx::parallel::util
