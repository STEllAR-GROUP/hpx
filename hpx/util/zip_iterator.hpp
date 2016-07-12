//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ZIP_ITERATOR_MAY_29_2014_0852PM)
#define HPX_UTIL_ZIP_ITERATOR_MAY_29_2014_0852PM

#include <hpx/config.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/functional/segmented_iterator_helpers.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/iterator/iterator_facade.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct zip_iterator_value;

        template <typename ...Ts>
        struct zip_iterator_value<tuple<Ts...> >
        {
            typedef tuple<typename std::iterator_traits<Ts>::value_type...> type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct zip_iterator_reference;

        template <typename ...Ts>
        struct zip_iterator_reference<tuple<Ts...> >
        {
            typedef tuple<typename std::iterator_traits<Ts>::reference...> type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename U>
        struct zip_iterator_category_impl
        {
            static_assert(
                sizeof(T) == 0 && sizeof(U) == 0,
                "unknown combination of iterator categories");
        };

        // random_access_iterator_tag
        template <>
        struct zip_iterator_category_impl<
            std::random_access_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::random_access_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::random_access_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::bidirectional_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::random_access_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::forward_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::random_access_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::input_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        // bidirectional_iterator_tag
        template <>
        struct zip_iterator_category_impl<
            std::bidirectional_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::bidirectional_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::forward_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::bidirectional_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::input_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        // forward_iterator_tag
        template <>
        struct zip_iterator_category_impl<
            std::forward_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::input_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::forward_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        // input_iterator_tag
        template <>
        struct zip_iterator_category_impl<
            std::input_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple, typename Enable = void>
        struct zip_iterator_category;

        template <typename T>
        struct zip_iterator_category<
            tuple<T>
          , typename std::enable_if<
                tuple_size<tuple<T> >::value == 1
            >::type
        >
        {
            typedef typename std::iterator_traits<T>::iterator_category type;
        };

        template <typename T, typename U>
        struct zip_iterator_category<
            tuple<T, U>
          , typename std::enable_if<
                tuple_size<tuple<T, U> >::value == 2
            >::type
        > : zip_iterator_category_impl<
                typename std::iterator_traits<T>::iterator_category
              , typename std::iterator_traits<U>::iterator_category
            >
        {};

        template <typename T, typename U, typename ...Tail>
        struct zip_iterator_category<
            tuple<T, U, Tail...>
          , typename std::enable_if<
                (tuple_size<tuple<T, U, Tail...> >::value > 2)
            >::type
        > : zip_iterator_category_impl<
                typename zip_iterator_category_impl<
                    typename std::iterator_traits<T>::iterator_category
                  , typename std::iterator_traits<U>::iterator_category
                >::type
              , typename zip_iterator_category<tuple<Tail...> >::type
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct dereference_iterator;

        template <typename ...Ts>
        struct dereference_iterator<tuple<Ts...> >
        {
            template <std::size_t ...Is>
            HPX_HOST_DEVICE
            static typename zip_iterator_reference<
                tuple<Ts...>
            >::type call(detail::pack_c<std::size_t, Is...>,
                tuple<Ts...> const& iterators)
            {
                return util::forward_as_tuple(*util::get<Is>(iterators)...);
            }
        };

        struct increment_iterator
        {
            template <typename T>
            HPX_HOST_DEVICE
            void operator()(T& iter) const
            {
                ++iter;
            }
        };

        struct decrement_iterator
        {
            template <typename T>
            HPX_HOST_DEVICE
            void operator()(T& iter) const
            {
                --iter;
            }
        };

        struct advance_iterator
        {
            explicit advance_iterator(std::ptrdiff_t n) : n_(n) {}

            template <typename T>
            HPX_HOST_DEVICE
            void operator()(T& iter) const
            {
                iter += n_;
            }

            std::ptrdiff_t n_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple, typename Derived>
        class zip_iterator_base
          : public boost::iterator_facade<
                Derived
              , typename zip_iterator_value<IteratorTuple>::type
              , typename zip_iterator_category<IteratorTuple>::type
              , typename zip_iterator_reference<IteratorTuple>::type
            >
        {
            typedef
                boost::iterator_facade<
                    zip_iterator_base<IteratorTuple, Derived>
                  , typename zip_iterator_value<IteratorTuple>::type
                  , typename zip_iterator_category<IteratorTuple>::type
                  , typename zip_iterator_reference<IteratorTuple>::type
                >
                base_type;

        public:
            HPX_HOST_DEVICE zip_iterator_base() {}

            HPX_HOST_DEVICE
            zip_iterator_base(IteratorTuple const& iterators)
              : iterators_(iterators)
            {}
            HPX_HOST_DEVICE
            zip_iterator_base(IteratorTuple && iterators)
              : iterators_(std::move(iterators))
            {}

            typedef IteratorTuple iterator_tuple_type;

            HPX_HOST_DEVICE iterator_tuple_type get_iterator_tuple()
            {
                return iterators_;
            }
            HPX_HOST_DEVICE iterator_tuple_type const& get_iterator_tuple() const
            {
                return iterators_;
            }

        private:
            friend class boost::iterator_core_access;

            HPX_HOST_DEVICE bool equal(zip_iterator_base const& other) const
            {
                return iterators_ == other.iterators_;
            }

            HPX_HOST_DEVICE typename base_type::reference dereference() const
            {
                return dereference_iterator<IteratorTuple>::call(
                    typename detail::make_index_pack<
                        util::tuple_size<IteratorTuple>::value
                    >::type(), iterators_);
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
                return util::get<0>(other.iterators_) - util::get<0>(iterators_);
            }

        private:
            template <typename F, std::size_t ...Is>
            HPX_HOST_DEVICE
            void apply(F&& f, detail::pack_c<std::size_t, Is...>)
            {
                int const _sequencer[]= {
                    ((f(util::get<Is>(iterators_))), 0)...
                };
                (void)_sequencer;
            }

            template <typename F>
            HPX_HOST_DEVICE void apply(F&& f)
            {
                return apply(std::forward<F>(f),
                    detail::make_index_pack<
                        util::tuple_size<IteratorTuple>::value
                    >());
            }

        private:
            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned)
            {
                ar & iterators_;
            }

        private:
            IteratorTuple iterators_;
        };
    }

    template <typename ...Ts>
    class zip_iterator
      : public detail::zip_iterator_base<
            tuple<Ts...>, zip_iterator<Ts...>
        >
    {
        typedef detail::zip_iterator_base<
                tuple<Ts...>, zip_iterator<Ts...>
            > base_type;

    public:
        HPX_HOST_DEVICE zip_iterator() : base_type() {}

        HPX_HOST_DEVICE explicit zip_iterator(Ts const&... vs)
          : base_type(util::tie(vs...))
        {}

        HPX_HOST_DEVICE explicit zip_iterator(tuple<Ts...> && t)
          : base_type(std::move(t))
        {}

        HPX_HOST_DEVICE zip_iterator(zip_iterator const& other)
          : base_type(other)
        {}

        HPX_HOST_DEVICE zip_iterator(zip_iterator&& other)
          : base_type(std::move(other))
        {}

        HPX_HOST_DEVICE zip_iterator& operator=(zip_iterator const& other)
        {
            base_type::operator=(other);
            return *this;
        }
        HPX_HOST_DEVICE zip_iterator& operator=(zip_iterator && other)
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        template <typename ... Ts_>
        HPX_HOST_DEVICE
        typename std::enable_if<
                detail::are_tuples_compatible_not_same<
                    typename zip_iterator::iterator_tuple_type,
                    typename zip_iterator<Ts_...>::iterator_tuple_type&&
                >::value,
                zip_iterator&
            >::type
        operator=(zip_iterator<Ts_...> const& other)
        {
            base_type::operator=(base_type(other.get_iterator_tuple()));
            return *this;
        }
        template <typename ... Ts_>
        HPX_HOST_DEVICE
        typename std::enable_if<
                detail::are_tuples_compatible_not_same<
                    typename zip_iterator::iterator_tuple_type,
                    typename zip_iterator<Ts_...>::iterator_tuple_type&&
                >::value,
                zip_iterator&
            >::type
        operator=(zip_iterator<Ts_...> && other)
        {
            base_type::operator=(base_type(std::move(other.get_iterator_tuple())));
            return *this;
        }
    };

    template <typename ...Ts>
    class zip_iterator<tuple<Ts...> >
      : public detail::zip_iterator_base<
            tuple<Ts...>, zip_iterator<tuple<Ts...> >
        >
    {
        typedef detail::zip_iterator_base<
                tuple<Ts...>, zip_iterator<tuple<Ts...> >
            > base_type;

    public:
        HPX_HOST_DEVICE zip_iterator() : base_type() {}

        HPX_HOST_DEVICE explicit zip_iterator(Ts const&... vs)
          : base_type(util::tie(vs...))
        {}

        HPX_HOST_DEVICE explicit zip_iterator(tuple<Ts...> && t)
          : base_type(std::move(t))
        {}

        HPX_HOST_DEVICE zip_iterator(zip_iterator const& other)
          : base_type(other)
        {}

        HPX_HOST_DEVICE zip_iterator(zip_iterator&& other)
          : base_type(std::move(other))
        {}

        HPX_HOST_DEVICE zip_iterator& operator=(zip_iterator const& other)
        {
            base_type::operator=(other);
            return *this;
        }
        HPX_HOST_DEVICE zip_iterator& operator=(zip_iterator && other)
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        template <typename ... Ts_>
        HPX_HOST_DEVICE
        typename std::enable_if<
                detail::are_tuples_compatible_not_same<
                    typename zip_iterator::iterator_tuple_type,
                    typename zip_iterator<Ts_...>::iterator_tuple_type&&
                >::value,
                zip_iterator&
            >::type
        operator=(zip_iterator<Ts_...> const& other)
        {
            base_type::operator=(base_type(other.get_iterator_tuple()));
            return *this;
        }
        template <typename ... Ts_>
        HPX_HOST_DEVICE
        typename std::enable_if<
                detail::are_tuples_compatible_not_same<
                    typename zip_iterator::iterator_tuple_type,
                    typename zip_iterator<Ts_...>::iterator_tuple_type&&
                >::value,
                zip_iterator&
            >::type
        operator=(zip_iterator<Ts_...> && other)
        {
            base_type::operator=(base_type(std::move(other.get_iterator_tuple())));
            return *this;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ...Ts>
    HPX_HOST_DEVICE zip_iterator<typename decay<Ts>::type...>
    make_zip_iterator(Ts&&... vs)
    {
        typedef zip_iterator<typename decay<Ts>::type...> result_type;

        return result_type(std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    struct zip_iterator_category
      : detail::zip_iterator_category<typename ZipIter::iterator_tuple_type>
    {};
}}

namespace hpx { namespace traits
{
    namespace functional
    {
        ///////////////////////////////////////////////////////////////////////
        struct get_raw_iterator
        {
            template <typename Iterator>
            struct apply
            {
                template <typename SegIter>
                typename segmented_iterator_traits<Iterator>::local_raw_iterator
                operator()(SegIter iter) const
                {
                    return iter.local();
                }
            };
        };

        struct get_remote_iterator
        {
            template <typename Iterator>
            struct apply
            {
                template <typename SegIter>
                typename segmented_iterator_traits<Iterator>::local_iterator
                operator()(SegIter iter) const
                {
                    return iter.remote();
                }
            };
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename T>
        struct element_result_of : util::result_of<F(T)> {};

        template <typename F, typename Iter>
        struct lift_zipped_iterators;

        template <typename F, typename ...Ts>
        struct lift_zipped_iterators<F, util::zip_iterator<Ts...> >
        {
            typedef typename util::zip_iterator<
                    Ts...
                >::iterator_tuple_type tuple_type;
            typedef util::tuple<
                    typename element_result_of<
                        typename F::template apply<Ts>, Ts
                    >::type...
                > result_type;

            template <std::size_t ...Is, typename ...Ts_>
            static result_type
            call(util::detail::pack_c<std::size_t, Is...>,
                util::tuple<Ts_...> const& t)
            {
                return util::make_tuple(
                    typename F::template apply<Ts>()(util::get<Is>(t))...);
            }

            template <typename ...Ts_>
            static result_type
            call(util::zip_iterator<Ts_...> const& iter)
            {
                using hpx::util::detail::make_index_pack;
                return call(typename make_index_pack<sizeof...(Ts)>::type(),
                    iter.get_iterator_tuple());
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // A zip_iterator represents a segmented iterator if all of the zipped
    // iterators are segmented iterators themselves.
    template <typename ...Ts>
    struct segmented_iterator_traits<
        util::zip_iterator<Ts...>,
        typename std::enable_if<
            util::detail::all_of<
                typename segmented_iterator_traits<Ts>::is_segmented_iterator...
            >::value
        >::type>
    {
        typedef std::true_type is_segmented_iterator;

        typedef util::zip_iterator<Ts...> iterator;
        typedef util::zip_iterator<
                typename segmented_iterator_traits<Ts>::segment_iterator...
            > segment_iterator;
        typedef util::zip_iterator<
                typename segmented_iterator_traits<Ts>::local_segment_iterator...
            > local_segment_iterator;
        typedef util::zip_iterator<
                typename segmented_iterator_traits<Ts>::local_iterator...
            > local_iterator;
        typedef util::zip_iterator<
                typename segmented_iterator_traits<Ts>::local_raw_iterator...
            > local_raw_iterator;

        //  Conceptually this function is supposed to denote which segment
        //  the iterator is currently pointing to (i.e. just global iterator).
        static segment_iterator segment(iterator iter)
        {
            return segment_iterator(
                functional::lift_zipped_iterators<
                        util::functional::segmented_iterator_segment, iterator
                    >::call(iter));
        }

        //  This function should specify which is the current segment and
        //  the exact position to which local iterator is pointing.
        static local_iterator local(iterator iter)
        {
            return local_iterator(
                functional::lift_zipped_iterators<
                        util::functional::segmented_iterator_local, iterator
                    >::call(iter));
        }

        //  This function should specify the local iterator which is at the
        //  beginning of the partition.
        static local_iterator begin(segment_iterator const& iter)
        {
            return local_iterator(
                functional::lift_zipped_iterators<
                        util::functional::segmented_iterator_begin, iterator
                    >::call(iter));
        }

        //  This function should specify the local iterator which is at the
        //  end of the partition.
        static local_iterator end(segment_iterator const& iter)
        {
            return local_iterator(
                functional::lift_zipped_iterators<
                        util::functional::segmented_iterator_end, iterator
                    >::call(iter));
        }

        //  This function should specify the local iterator which is at the
        //  beginning of the partition data.
        static local_raw_iterator begin(local_segment_iterator const& seg_iter)
        {
            return local_raw_iterator(
                functional::lift_zipped_iterators<
                        util::functional::segmented_iterator_local_begin,
                        iterator
                    >::call(seg_iter));
        }

        //  This function should specify the local iterator which is at the
        //  end of the partition data.
        static local_raw_iterator end(local_segment_iterator const& seg_iter)
        {
            return local_raw_iterator(
                functional::lift_zipped_iterators<
                        util::functional::segmented_iterator_local_end,
                        iterator
                    >::call(seg_iter));
        }

        // Extract the base id for the segment referenced by the given segment
        // iterator.
        static naming::id_type get_id(segment_iterator const& iter)
        {
            typedef typename util::tuple_element<
                    0, typename iterator::iterator_tuple_type
                >::type first_base_iterator;
            typedef segmented_iterator_traits<first_base_iterator> traits;

            return traits::get_id(util::get<0>(iter.get_iterator_tuple()));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ...Ts>
    struct segmented_local_iterator_traits<
        util::zip_iterator<Ts...>,
        typename std::enable_if<
            util::detail::all_of<
                typename segmented_local_iterator_traits<Ts>
                ::is_segmented_local_iterator...
            >::value
        >::type>
    {
        typedef std::true_type is_segmented_local_iterator;

        typedef util::zip_iterator<
                typename segmented_local_iterator_traits<Ts>::iterator...
            > iterator;
        typedef util::zip_iterator<Ts...> local_iterator;
        typedef util::zip_iterator<
                typename segmented_local_iterator_traits<Ts>::local_raw_iterator...
            > local_raw_iterator;

        // Extract base iterator from local_iterator
        static local_raw_iterator local(local_iterator const& iter)
        {
            return local_raw_iterator(
                functional::lift_zipped_iterators<
                        functional::get_raw_iterator, iterator
                    >::call(iter));
        }

        // Construct remote local_iterator from local_raw_iterator
        static local_iterator remote(local_raw_iterator const& iter)
        {
            return local_iterator(
                functional::lift_zipped_iterators<
                        functional::get_remote_iterator, iterator
                    >::call(iter));
        }
    };
}}

#endif
