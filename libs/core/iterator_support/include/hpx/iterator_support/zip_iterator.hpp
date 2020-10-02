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
            typedef hpx::tuple<typename std::iterator_traits<Ts>::value_type...>
                type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct zip_iterator_reference;

        template <typename... Ts>
        struct zip_iterator_reference<hpx::tuple<Ts...>>
        {
            typedef hpx::tuple<typename std::iterator_traits<Ts>::reference...>
                type;
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
            typedef std::random_access_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::random_access_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::bidirectional_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::random_access_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::forward_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::random_access_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::input_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        // bidirectional_iterator_tag
        template <>
        struct zip_iterator_category_impl<std::bidirectional_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::bidirectional_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::forward_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::bidirectional_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::input_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        // forward_iterator_tag
        template <>
        struct zip_iterator_category_impl<std::forward_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::input_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::forward_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        // input_iterator_tag
        template <>
        struct zip_iterator_category_impl<std::input_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple, typename Enable = void>
        struct zip_iterator_category;

        template <typename T>
        struct zip_iterator_category<hpx::tuple<T>,
            typename std::enable_if<hpx::tuple_size<hpx::tuple<T>>::value ==
                1>::type>
        {
            typedef typename std::iterator_traits<T>::iterator_category type;
        };

        template <typename T, typename U>
        struct zip_iterator_category<hpx::tuple<T, U>,
            typename std::enable_if<hpx::tuple_size<hpx::tuple<T, U>>::value ==
                2>::type>
          : zip_iterator_category_impl<
                typename std::iterator_traits<T>::iterator_category,
                typename std::iterator_traits<U>::iterator_category>
        {
        };

        template <typename T, typename U, typename... Tail>
        struct zip_iterator_category<hpx::tuple<T, U, Tail...>,
            typename std::enable_if<(
                hpx::tuple_size<hpx::tuple<T, U, Tail...>>::value > 2)>::type>
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
            HPX_HOST_DEVICE static
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
            HPX_HOST_DEVICE void operator()(T& iter) const
            {
                ++iter;
            }
        };

        struct decrement_iterator
        {
            template <typename T>
            HPX_HOST_DEVICE void operator()(T& iter) const
            {
                --iter;
            }
        };

        struct advance_iterator
        {
            explicit advance_iterator(std::ptrdiff_t n)
              : n_(n)
            {
            }

            template <typename T>
            HPX_HOST_DEVICE void operator()(T& iter) const
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
            typedef hpx::util::iterator_facade<
                zip_iterator_base<IteratorTuple, Derived>,
                typename zip_iterator_value<IteratorTuple>::type,
                typename zip_iterator_category<IteratorTuple>::type,
                typename zip_iterator_reference<IteratorTuple>::type>
                base_type;

        public:
            HPX_HOST_DEVICE zip_iterator_base() {}

            HPX_HOST_DEVICE
            zip_iterator_base(IteratorTuple const& iterators)
              : iterators_(iterators)
            {
            }
            HPX_HOST_DEVICE
            zip_iterator_base(IteratorTuple&& iterators)
              : iterators_(std::move(iterators))
            {
            }

            typedef IteratorTuple iterator_tuple_type;

            HPX_HOST_DEVICE iterator_tuple_type get_iterator_tuple() const
            {
                return iterators_;
            }

        private:
            friend class hpx::util::iterator_core_access;

            HPX_HOST_DEVICE bool equal(zip_iterator_base const& other) const
            {
                return iterators_ == other.iterators_;
            }

            HPX_HOST_DEVICE typename base_type::reference dereference() const
            {
                return dereference_iterator<IteratorTuple>::call(
                    typename util::make_index_pack<
                        hpx::tuple_size<IteratorTuple>::value>::type(),
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
                return apply(std::forward<F>(f),
                    util::make_index_pack<
                        hpx::tuple_size<IteratorTuple>::value>());
            }

        private:
            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned)
            {
                ar& iterators_;
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

        typedef detail::zip_iterator_base<hpx::tuple<Ts...>,
            zip_iterator<Ts...>>
            base_type;

    public:
        HPX_HOST_DEVICE zip_iterator()
          : base_type()
        {
        }

        HPX_HOST_DEVICE explicit zip_iterator(Ts const&... vs)
          : base_type(hpx::tie(vs...))
        {
        }

        HPX_HOST_DEVICE explicit zip_iterator(hpx::tuple<Ts...>&& t)
          : base_type(std::move(t))
        {
        }

        HPX_HOST_DEVICE zip_iterator(zip_iterator const& other)
          : base_type(other)
        {
        }

        HPX_HOST_DEVICE zip_iterator(zip_iterator&& other)
          : base_type(std::move(other))
        {
        }

        HPX_HOST_DEVICE zip_iterator& operator=(zip_iterator const& other)
        {
            base_type::operator=(other);
            return *this;
        }
        HPX_HOST_DEVICE zip_iterator& operator=(zip_iterator&& other)
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        template <typename... Ts_>
        HPX_HOST_DEVICE typename std::enable_if<
            std::is_assignable<typename zip_iterator::iterator_tuple_type&,
                typename zip_iterator<Ts_...>::iterator_tuple_type&&>::value,
            zip_iterator&>::type
        operator=(zip_iterator<Ts_...> const& other)
        {
            base_type::operator=(base_type(other.get_iterator_tuple()));
            return *this;
        }
        template <typename... Ts_>
        HPX_HOST_DEVICE typename std::enable_if<
            std::is_assignable<typename zip_iterator::iterator_tuple_type&,
                typename zip_iterator<Ts_...>::iterator_tuple_type&&>::value,
            zip_iterator&>::type
        operator=(zip_iterator<Ts_...>&& other)
        {
            base_type::operator=(
                base_type(std::move(other.get_iterator_tuple())));
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

        typedef detail::zip_iterator_base<hpx::tuple<Ts...>,
            zip_iterator<hpx::tuple<Ts...>>>
            base_type;

    public:
        HPX_HOST_DEVICE zip_iterator()
          : base_type()
        {
        }

        HPX_HOST_DEVICE explicit zip_iterator(Ts const&... vs)
          : base_type(hpx::tie(vs...))
        {
        }

        HPX_HOST_DEVICE explicit zip_iterator(hpx::tuple<Ts...>&& t)
          : base_type(std::move(t))
        {
        }

        HPX_HOST_DEVICE zip_iterator(zip_iterator const& other)
          : base_type(other)
        {
        }

        HPX_HOST_DEVICE zip_iterator(zip_iterator&& other)
          : base_type(std::move(other))
        {
        }

        HPX_HOST_DEVICE zip_iterator& operator=(zip_iterator const& other)
        {
            base_type::operator=(other);
            return *this;
        }
        HPX_HOST_DEVICE zip_iterator& operator=(zip_iterator&& other)
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        template <typename... Ts_>
        HPX_HOST_DEVICE typename std::enable_if<
            std::is_assignable<typename zip_iterator::iterator_tuple_type&,
                typename zip_iterator<Ts_...>::iterator_tuple_type&&>::value,
            zip_iterator&>::type
        operator=(zip_iterator<Ts_...> const& other)
        {
            base_type::operator=(base_type(other.get_iterator_tuple()));
            return *this;
        }
        template <typename... Ts_>
        HPX_HOST_DEVICE typename std::enable_if<
            std::is_assignable<typename zip_iterator::iterator_tuple_type&,
                typename zip_iterator<Ts_...>::iterator_tuple_type&&>::value,
            zip_iterator&>::type
        operator=(zip_iterator<Ts_...>&& other)
        {
            base_type::operator=(
                base_type(std::move(other.get_iterator_tuple())));
            return *this;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    HPX_HOST_DEVICE zip_iterator<typename decay<Ts>::type...> make_zip_iterator(
        Ts&&... vs)
    {
        typedef zip_iterator<typename decay<Ts>::type...> result_type;

        return result_type(std::forward<Ts>(vs)...);
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
            typedef typename util::zip_iterator<Ts...>::iterator_tuple_type
                tuple_type;
            typedef hpx::tuple<typename element_result_of<
                typename F::template apply<Ts>, Ts>::type...>
                result_type;

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
                using hpx::util::make_index_pack;
                return call(typename make_index_pack<sizeof...(Ts)>::type(),
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
