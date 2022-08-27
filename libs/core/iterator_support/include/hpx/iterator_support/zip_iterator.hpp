//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/iterator_support/iterator_facade.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct zip_iterator_value;

        template <typename... Ts>
        struct zip_iterator_value<hpx::tuple<Ts...>>
        {
            using type =
                hpx::tuple<typename std::iterator_traits<Ts>::value_type...>;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct zip_iterator_reference;

        template <typename... Ts>
        struct zip_iterator_reference<hpx::tuple<Ts...>>
        {
            using type =
                hpx::tuple<typename std::iterator_traits<Ts>::reference...>;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename U>
        struct zip_iterator_category_impl
        {
            static_assert(sizeof(T) == 0 && sizeof(U) == 0,
                "unknown combination of iterator categories");
        };

        // random_access_iterator_tag
        template <>
        struct zip_iterator_category_impl<std::random_access_iterator_tag,
            std::random_access_iterator_tag>
        {
            using type = std::random_access_iterator_tag;
        };

        template <>
        struct zip_iterator_category_impl<std::random_access_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            using type = std::bidirectional_iterator_tag;
        };

        template <>
        struct zip_iterator_category_impl<std::bidirectional_iterator_tag,
            std::random_access_iterator_tag>
        {
            using type = std::bidirectional_iterator_tag;
        };

        template <>
        struct zip_iterator_category_impl<std::random_access_iterator_tag,
            std::forward_iterator_tag>
        {
            using type = std::forward_iterator_tag;
        };

        template <>
        struct zip_iterator_category_impl<std::forward_iterator_tag,
            std::random_access_iterator_tag>
        {
            using type = std::forward_iterator_tag;
        };

        template <>
        struct zip_iterator_category_impl<std::random_access_iterator_tag,
            std::input_iterator_tag>
        {
            using type = std::input_iterator_tag;
        };

        template <>
        struct zip_iterator_category_impl<std::input_iterator_tag,
            std::random_access_iterator_tag>
        {
            using type = std::input_iterator_tag;
        };

        // bidirectional_iterator_tag
        template <>
        struct zip_iterator_category_impl<std::bidirectional_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            using type = std::bidirectional_iterator_tag;
        };

        template <>
        struct zip_iterator_category_impl<std::bidirectional_iterator_tag,
            std::forward_iterator_tag>
        {
            using type = std::forward_iterator_tag;
        };

        template <>
        struct zip_iterator_category_impl<std::forward_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            using type = std::forward_iterator_tag;
        };

        template <>
        struct zip_iterator_category_impl<std::bidirectional_iterator_tag,
            std::input_iterator_tag>
        {
            using type = std::input_iterator_tag;
        };

        template <>
        struct zip_iterator_category_impl<std::input_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            using type = std::input_iterator_tag;
        };

        // forward_iterator_tag
        template <>
        struct zip_iterator_category_impl<std::forward_iterator_tag,
            std::forward_iterator_tag>
        {
            using type = std::forward_iterator_tag;
        };

        template <>
        struct zip_iterator_category_impl<std::input_iterator_tag,
            std::forward_iterator_tag>
        {
            using type = std::input_iterator_tag;
        };

        template <>
        struct zip_iterator_category_impl<std::forward_iterator_tag,
            std::input_iterator_tag>
        {
            using type = std::input_iterator_tag;
        };

        // input_iterator_tag
        template <>
        struct zip_iterator_category_impl<std::input_iterator_tag,
            std::input_iterator_tag>
        {
            using type = std::input_iterator_tag;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple, typename Enable = void>
        struct zip_iterator_category;

        template <typename T>
        struct zip_iterator_category<hpx::tuple<T>,
            std::enable_if_t<hpx::tuple_size<hpx::tuple<T>>::value == 1>>
        {
            using type = typename std::iterator_traits<T>::iterator_category;
        };

        template <typename T, typename U>
        struct zip_iterator_category<hpx::tuple<T, U>,
            std::enable_if_t<hpx::tuple_size<hpx::tuple<T, U>>::value == 2>>
          : zip_iterator_category_impl<
                typename std::iterator_traits<T>::iterator_category,
                typename std::iterator_traits<U>::iterator_category>
        {
        };

        template <typename T, typename U, typename... Tail>
        struct zip_iterator_category<hpx::tuple<T, U, Tail...>,
            std::enable_if_t<(
                hpx::tuple_size<hpx::tuple<T, U, Tail...>>::value > 2)>>
          : zip_iterator_category_impl<
                typename zip_iterator_category_impl<
                    typename std::iterator_traits<T>::iterator_category,
                    typename std::iterator_traits<U>::iterator_category>::type,
                typename zip_iterator_category<hpx::tuple<Tail...>>::type>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct dereference_iterator;

        template <typename... Ts>
        struct dereference_iterator<hpx::tuple<Ts...>>
        {
            template <std::size_t... Is>
            HPX_HOST_DEVICE static constexpr
                typename zip_iterator_reference<hpx::tuple<Ts...>>::type
                call(
                    util::index_pack<Is...>, hpx::tuple<Ts...> const& iterators)
            {
                return hpx::forward_as_tuple(*hpx::get<Is>(iterators)...);
            }
        };

        struct increment_iterator
        {
            template <typename T>
            HPX_HOST_DEVICE constexpr void operator()(T& iter) const
                noexcept(noexcept(++std::declval<T&>()))
            {
                ++iter;
            }
        };

        struct decrement_iterator
        {
            template <typename T>
            HPX_HOST_DEVICE constexpr void operator()(T& iter) const
                noexcept(noexcept(--std::declval<T&>()))
            {
                --iter;
            }
        };

        struct advance_iterator
        {
            explicit constexpr advance_iterator(std::ptrdiff_t n) noexcept
              : n_(n)
            {
            }

            template <typename T>
            HPX_HOST_DEVICE constexpr void operator()(T& iter) const noexcept(
                noexcept(std::declval<T&>() += std::declval<std::ptrdiff_t>()))
            {
                iter += n_;
            }

            std::ptrdiff_t n_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple, typename Derived>
        class zip_iterator_base
          : public hpx::util::iterator_facade<Derived,
                typename zip_iterator_value<IteratorTuple>::type,
                typename zip_iterator_category<IteratorTuple>::type,
                typename zip_iterator_reference<IteratorTuple>::type>
        {
            using base_type = hpx::util::iterator_facade<
                zip_iterator_base<IteratorTuple, Derived>,
                typename zip_iterator_value<IteratorTuple>::type,
                typename zip_iterator_category<IteratorTuple>::type,
                typename zip_iterator_reference<IteratorTuple>::type>;

        public:
            HPX_HOST_DEVICE constexpr zip_iterator_base() noexcept {}

            HPX_HOST_DEVICE
            constexpr zip_iterator_base(IteratorTuple const& iterators)
              : iterators_(iterators)
            {
            }
            HPX_HOST_DEVICE
            constexpr zip_iterator_base(IteratorTuple&& iterators) noexcept
              : iterators_(HPX_MOVE(iterators))
            {
            }

            using iterator_tuple_type = IteratorTuple;

            HPX_HOST_DEVICE constexpr iterator_tuple_type const&
            get_iterator_tuple() const& noexcept
            {
                return iterators_;
            }

            HPX_HOST_DEVICE constexpr iterator_tuple_type&&
            get_iterator_tuple() && noexcept
            {
                return HPX_MOVE(iterators_);
            }

        private:
            friend class hpx::util::iterator_core_access;

            HPX_HOST_DEVICE constexpr bool equal(
                zip_iterator_base const& other) const
                noexcept(noexcept(std::declval<IteratorTuple>() ==
                    std::declval<IteratorTuple>()))
            {
                return iterators_ == other.iterators_;
            }

            HPX_HOST_DEVICE constexpr typename base_type::reference
            dereference() const
            {
                return dereference_iterator<IteratorTuple>::call(
                    util::make_index_pack_t<
                        hpx::tuple_size<IteratorTuple>::value>(),
                    iterators_);
            }

            HPX_HOST_DEVICE void increment()
            {
                this->apply(increment_iterator());
            }

            HPX_HOST_DEVICE void decrement()
            {
                this->apply(decrement_iterator());
            }

            HPX_HOST_DEVICE void advance(std::ptrdiff_t n)
            {
                this->apply(advance_iterator(n));
            }

            HPX_HOST_DEVICE
            std::ptrdiff_t distance_to(zip_iterator_base const& other) const
            {
                return hpx::get<0>(other.iterators_) - hpx::get<0>(iterators_);
            }

        private:
            template <typename F, std::size_t... Is>
            HPX_HOST_DEVICE void apply(F&& f, util::index_pack<Is...>)
            {
                int const _sequencer[] = {
                    ((f(hpx::get<Is>(iterators_))), 0)...};
                (void) _sequencer;
            }

            template <typename F>
            HPX_HOST_DEVICE void apply(F&& f)
            {
                return apply(HPX_FORWARD(F, f),
                    util::make_index_pack<
                        hpx::tuple_size<IteratorTuple>::value>());
            }

        private:
            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned)
            {
                // clang-format off
                ar & iterators_;
                // clang-format on
            }

        private:
            IteratorTuple iterators_;
        };
    }    // namespace detail

    template <typename... Ts>
    class zip_iterator
      : public detail::zip_iterator_base<hpx::tuple<Ts...>, zip_iterator<Ts...>>
    {
        static_assert(
            sizeof...(Ts) != 0, "zip_iterator must wrap at least one iterator");

        using base_type =
            detail::zip_iterator_base<hpx::tuple<Ts...>, zip_iterator<Ts...>>;

    public:
        HPX_HOST_DEVICE constexpr zip_iterator() noexcept
          : base_type()
        {
        }

        HPX_HOST_DEVICE explicit constexpr zip_iterator(
            Ts const&... vs) noexcept
          : base_type(hpx::tie(vs...))
        {
        }

        HPX_HOST_DEVICE explicit constexpr zip_iterator(
            hpx::tuple<Ts...>&& t) noexcept
          : base_type(HPX_MOVE(t))
        {
        }

        HPX_HOST_DEVICE constexpr zip_iterator(zip_iterator const& other)
          : base_type(other)
        {
        }

        HPX_HOST_DEVICE constexpr zip_iterator(zip_iterator&& other) noexcept
          : base_type(HPX_MOVE(other))
        {
        }

        HPX_HOST_DEVICE zip_iterator& operator=(zip_iterator const& other)
        {
            base_type::operator=(other);
            return *this;
        }
        HPX_HOST_DEVICE zip_iterator& operator=(zip_iterator&& other) noexcept
        {
            base_type::operator=(HPX_MOVE(other));
            return *this;
        }

        template <typename... Ts_>
        HPX_HOST_DEVICE std::enable_if_t<
            std::is_assignable_v<typename zip_iterator::iterator_tuple_type&,
                typename zip_iterator<Ts_...>::iterator_tuple_type&&>,
            zip_iterator&>
        operator=(zip_iterator<Ts_...> const& other)
        {
            base_type::operator=(base_type(other.get_iterator_tuple()));
            return *this;
        }

        template <typename... Ts_>
        HPX_HOST_DEVICE std::enable_if_t<
            std::is_assignable_v<typename zip_iterator::iterator_tuple_type&,
                typename zip_iterator<Ts_...>::iterator_tuple_type&&>,
            zip_iterator&>
        operator=(zip_iterator<Ts_...>&& other) noexcept
        {
            base_type::operator=(
                base_type(HPX_MOVE(other).get_iterator_tuple()));
            return *this;
        }
    };

    template <typename... Ts>
    class zip_iterator<hpx::tuple<Ts...>>
      : public detail::zip_iterator_base<hpx::tuple<Ts...>,
            zip_iterator<hpx::tuple<Ts...>>>
    {
        static_assert(
            sizeof...(Ts) != 0, "zip_iterator must wrap at least one iterator");

        using base_type = detail::zip_iterator_base<hpx::tuple<Ts...>,
            zip_iterator<hpx::tuple<Ts...>>>;

    public:
        HPX_HOST_DEVICE constexpr zip_iterator() noexcept
          : base_type()
        {
        }

        HPX_HOST_DEVICE explicit constexpr zip_iterator(
            Ts const&... vs) noexcept
          : base_type(hpx::tie(vs...))
        {
        }

        HPX_HOST_DEVICE explicit constexpr zip_iterator(
            hpx::tuple<Ts...>&& t) noexcept
          : base_type(HPX_MOVE(t))
        {
        }

        HPX_HOST_DEVICE constexpr zip_iterator(zip_iterator const& other)
          : base_type(other)
        {
        }

        HPX_HOST_DEVICE constexpr zip_iterator(zip_iterator&& other) noexcept
          : base_type(HPX_MOVE(other))
        {
        }

        HPX_HOST_DEVICE zip_iterator& operator=(zip_iterator const& other)
        {
            base_type::operator=(other);
            return *this;
        }

        HPX_HOST_DEVICE zip_iterator& operator=(zip_iterator&& other) noexcept
        {
            base_type::operator=(HPX_MOVE(other));
            return *this;
        }

        template <typename... Ts_>
        HPX_HOST_DEVICE std::enable_if_t<
            std::is_assignable_v<typename zip_iterator::iterator_tuple_type&,
                typename zip_iterator<Ts_...>::iterator_tuple_type&&>,
            zip_iterator&>
        operator=(zip_iterator<Ts_...> const& other)
        {
            base_type::operator=(base_type(other.get_iterator_tuple()));
            return *this;
        }

        template <typename... Ts_>
        HPX_HOST_DEVICE std::enable_if_t<
            std::is_assignable_v<typename zip_iterator::iterator_tuple_type&,
                typename zip_iterator<Ts_...>::iterator_tuple_type&&>,
            zip_iterator&>
        operator=(zip_iterator<Ts_...>&& other) noexcept
        {
            base_type::operator=(
                base_type(HPX_MOVE(other).get_iterator_tuple()));
            return *this;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    HPX_HOST_DEVICE constexpr zip_iterator<std::decay_t<Ts>...>
    make_zip_iterator(Ts&&... vs)
    {
        using result_type = zip_iterator<std::decay_t<Ts>...>;
        return result_type(HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    struct zip_iterator_category
      : detail::zip_iterator_category<typename ZipIter::iterator_tuple_type>
    {
    };
}}    // namespace hpx::util

namespace hpx { namespace traits {

    namespace functional {

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename T>
        struct element_result_of : util::invoke_result<F, T>
        {
        };

        template <typename F, typename Iter>
        struct lift_zipped_iterators;

        template <typename F, typename... Ts>
        struct lift_zipped_iterators<F, util::zip_iterator<Ts...>>
        {
            using tuple_type =
                typename util::zip_iterator<Ts...>::iterator_tuple_type;
            using result_type = hpx::tuple<typename element_result_of<
                typename F::template apply<Ts>, Ts>::type...>;

            template <std::size_t... Is, typename... Ts_>
            static result_type call(
                util::index_pack<Is...>, hpx::tuple<Ts_...> const& t)
            {
                return hpx::make_tuple(
                    typename F::template apply<Ts>()(hpx::get<Is>(t))...);
            }

            template <typename... Ts_>
            static result_type call(util::zip_iterator<Ts_...> const& iter)
            {
                using hpx::util::make_index_pack_t;
                return call(make_index_pack_t<sizeof...(Ts)>(),
                    iter.get_iterator_tuple());
            }
        };
    }    // namespace functional

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Iter>
    struct is_zip_iterator<hpx::util::zip_iterator<Iter...>> : std::true_type
    {
    };
}}    // namespace hpx::traits
