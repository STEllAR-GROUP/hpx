//  Copyright (c) 2016 Zahra Khatami
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PREFETCHING_LOOP)
#define HPX_PREFETCHING_LOOP

#include <hpx/config.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/traits/is_range.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

#if defined(HPX_HAVE_MM_PREFETCH)
#if defined(HPX_MSVC)
#include <mmintrin.h>
#endif
#if defined(HPX_GCC_VERSION)
#include <emmintrin.h>
#endif
#endif

namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {
        template <typename Itr, typename ...Ts>
        class prefetching_iterator
          : public std::iterator<std::random_access_iterator_tag, std::size_t>
        {
        public:
            typedef Itr base_iterator;

        private:
            hpx::util::tuple<Ts const& ...> rngs_;
            base_iterator base_;
            std::size_t chunk_size_;
            std::size_t range_size_;
            std::size_t idx_;

        public:
            explicit prefetching_iterator(std::size_t idx,
                    base_iterator base, std::size_t chunk_size,
                    std::size_t range_size,
                    hpx::util::tuple<Ts*...> const& rngs)
              : rngs_(rngs), base_(base), chunk_size_(chunk_size),
                range_size_(range_size), idx_(idx)
            {}

            typedef typename std::iterator<
                    std::random_access_iterator_tag, std::size_t
                >::difference_type difference_type;

            inline prefetching_iterator& operator+=(difference_type rhs)
            {
                idx_ += rhs * chunk_size_;
                std::advance(base_, rhs_ * chunk_size_);
                return *this;
            }
            inline prefetching_iterator& operator-=(difference_type rhs)
            {
                idx_ -= rhs * chunk_size_;
                std::advance(base_, -(rhs_ * chunk_size_));
                return *this;
            }
            inline prefetching_iterator& operator++()
            {
                idx_ += chunk_size_;
                std::advance(base_, chunk_size_);
                return *this;
            }
            inline prefetching_iterator& operator--()
            {
                idx_ -= chunk_size_;
                std::advance(base_, -chunk_size_);
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

            // FIXME: What happens if the distance is not divisible by the
            //        chunk_size? Should enforce this?
            inline difference_type
            operator-(const prefetching_iterator& rhs) const
            {
                return (idx_ - rhs.idx_) / chunk_size_;
            }
            inline prefetching_iterator
            operator+(difference_type rhs) const
            {
                return prefetching_iterator(idx_ + (rhs * chunk_size_),
                    base_ + (rhs * chunk_size_), chunk_size_, range_size_, rngs_);
            }

            inline prefetching_iterator
            operator-(difference_type rhs) const
            {
                return prefetching_iterator(idx_ - (rhs * chunk_size_),
                    base_ - (rhs * chunk_size_), chunk_size_, range_size_, rngs_);
            }

            friend inline prefetching_iterator
            operator+(const prefetching_iterator& lhs, difference_type rhs)
            {
                return rhs + lhs;
            }
            friend inline prefetching_iterator
            operator-(const prefetching_iterator& lhs, difference_type rhs)
            {
                return lhs - rhs;
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
            inline std::size_t & operator[](std::size_t rhs)
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
        template <typename Itr, typename ... Ts>
        struct prefetcher_context
        {
        private:
            Itr it_begin_;
            Itr it_end_;
            hpx::util::tuple<Ts const&...> rngs_;
            std::size_t chunk_size_;
            std::size_t range_size_;

        public:
            // FIXME: cache line size is probably platform dependent
            static HPX_CONSTEXPR_OR_CONST std::size_t cache_line_size = 64ull;

            prefetcher_context (Itr begin, Itr end,
                    hpx::util::tuple<Ts const&...> rngs,
                    std::size_t p_factor = 1)
              : it_begin_(begin), it_end_(end), rngs_(rngs),
                chunk_size_((p_factor * cache_line_size) /
                    sizeof(typename hpx::util::tuple_element<0, rngs>::type))
                range_size_(std::distance(begin, end))
            {}

            prefetching_iterator<Itr, Ts const&...> begin()
            {
                return prefetching_iterator<Itr, Ts const&...>(
                    0ull, it_begin_, chunk_size_, range_size_, rngs_);
            }

            prefetching_iterator<Itr, Ts const&...> end()
            {
                return prefetching_iterator<Itr, Ts const&...>(
                    range_size_, it_end_, chunk_size_, range_size_, rngs_);
            }
        };

        ///////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_MM_PREFETCH)
        template <typename T>
        HPX_FORCEINLINE void prefetch_address(T const& val)
        {
            _mm_prefetch((char*)&val, _MM_HINT_T0);
        }

        template <typename ... Ts, std::size_t ... Is>
        HPX_FORCEINLINE void
        prefetch_containers(hpx::util::tuple<Ts...> const& t,
            hpx::util::detail::pack_c<std::size_t, Is...>, std::size_t idx)
        {
            int const sequencer[] = {
                (prefetch_address(hpx::util::get<Is>(t)[idx])..., 0), 0
            };
            (void)sequencer;
        }
#else
        template <typename ... Ts, std::size_t ... Is>
        HPX_FORCEINLINE void
        prefetch_containers(hpx::util::tuple<Ts...> const& t,
            hpx::util::detail::pack_c<std::size_t, Is...>, std::size_t idx)
        {
            int const sequencer[] = {
                (hpx::util::get<Is>(t)[idx], 0)..., 0
            };
            (void)sequencer;
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename Itr, typename ... Ts>
        struct loop<detail::prefetching_iterator<Itr, Ts...> >
        {
            typedef detail::prefetching_iterator<Itr, Ts...> iterator_type;
            typedef typename iterator_type::base_iterator type;
            typedef typename hpx::util::detail::make_index_pack<
                    sizeof...(Ts)
                >::type index_pack_type;

            template <typename End, typename F>
            static iterator_type
            call(iterator_type it, End end, F && f)
            {
                for (/**/; it != end; ++it)
                {
                    Itr base = it.base_;
                    std::size_t j = it.idx_;

                    std::size_t last = (std::min)(it.idx_ + it.chunk_size_,
                        it.range_size_);

                    for (/**/; j != last; (void) ++j, ++base)
                        f(base);

                    if (j != it.range_size_)
                        prefetch_containers(it.rngs_, index_pack_type(), j);
                }
                return it;
            }

            template <typename End, typename CancelToken, typename F>
            static iterator_type
            call(iterator_type it, End end, CancelToken& tok, F && f)
            {
                for (/**/; it != end; ++it)
                {
                    if (tok.was_cancelled())
                        break;

                    Itr base = it.base_;
                    std::size_t j = it.idx_;

                    std::size_t last = (std::min)(it.idx_ + it.chunk_size_,
                        it.range_size_);

                    for (/**/; j != last; (void) ++j, ++base)
                        f(base);

                    if (j != it.range_size_)
                        prefetch_containers(it.rngs_, index_pack_type(), j);
                }
                return it;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Itr, typename ... Ts>
        struct loop_n<detail::prefetching_iterator<Itr, Ts...> >
        {
            typedef detail::prefetching_iterator<Itr, Ts...> iterator_type;
            typedef typename iterator_type::base_iterator type;
            typedef typename hpx::util::detail::make_index_pack<
                    sizeof...(Ts)
                >::type index_pack_type;

            template <typename F>
            static iterator_type 
            call(iterator_type it, std::size_t count, F && f)
            {
                for (/**/; count >= 0; (void) --count, ++it)
                {
                    Itr base = it.base_;
                    std::size_t j = it.idx_;

                    std::size_t last = (std::min)(it.idx_ + it.chunk_size_,
                        it.range_size_);

                    for (/**/; j != last; (void) ++j, ++base)
                        f(base);

                    if (j != it.range_size_)
                        prefetch_containers(it.rngs_, index_pack_type(), j);
                }
                return it;
            }

            template <typename CancelToken, typename F>
            static iterator_type
            call(iterator_type it, std::size_t count, CancelToken& tok, F && f)
            {
                for (/**/; count >= 0; (void) --count, ++it)
                {
                    if (tok.was_cancelled())
                        break;

                    Itr base = it.base_;
                    std::size_t j = it.idx_;

                    std::size_t last = (std::min)(it.idx_ + it.chunk_size_,
                        it.range_size_);

                    for (/**/; j != last; (void) ++j, ++base)
                        f(base);

                    if (j != it.range_size_)
                        prefetch_containers(it.rngs_, index_pack_type(), j);
                }
                return it;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // function to create a prefetcher_context
    template <typename Itr, typename ... Ts>
    prefetcher_context<Itr, Ts...>
    make_prefetcher_context(Itr base_begin, Itr base_end,
        std::size_t p_factor, Ts const& ... rngs)
    {
        static_assert(
            hpx::util::detail::all_of<hpx::traits::is_range<Ts>...>::value,
            "All variadic parameters have to represent ranges");

        return prefetcher_context<Itr, Ts...>(base_begin, base_end,
            hpx::util::forward_as_tuple(rngs...), p_factor);
    }
}}}

#endif

