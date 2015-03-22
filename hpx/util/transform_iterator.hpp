//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_TRANSFORM_ITERATOR_MAR_19_2015_0813AM)
#define HPX_UTIL_TRANSFORM_ITERATOR_MAR_19_2015_0813AM

#include <hpx/util/result_of.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/serialization/serialization.hpp>

#include <type_traits>
#include <iterator>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator, typename Transformer,
            typename Reference = void, typename Value = void>
    class transform_iterator;
// }}
//
// namespace hpx { namespace traits
// {
//     template <typename Transformer, typename Enable = void>
//     struct transform_iterator_transformer_traits
//     {
//         template <typename T>
//         struct result;
//
//         template <typename This, typename Transformer_, typename F>
//         struct result<This(Transformer_, F)>
//         {
//             typedef Transformer type;
//         };
//
//         template <typename F>
//         Transformer const&
//         operator()(Transformer const& t, F const&) const
//         {
//             return t;
//         }
//     };
// }}
//
// namespace hpx { namespace util
// {
    namespace detail
    {
        template <typename Iterator, typename Transformer, typename Reference,
            typename Value>
        struct transform_iterator_base
        {
            typedef typename std::conditional<
                    std::is_void<Reference>::value,
                    typename util::result_of<Transformer(Iterator)>::type,
                    Reference
                >::type reference_type;

            typedef typename std::conditional<
                    std::is_void<Value>::value,
                    typename std::remove_reference<reference_type>::type,
                    Value
                >::type value_type;

            typedef boost::iterator_adaptor<
                transform_iterator<Iterator, Transformer, Reference, Value>,
                Iterator, value_type, boost::use_default, value_type
            > type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // The main difference to boost::transform_iterator is that the transformer
    // function will be invoked with the iterator, not with the result of
    // dereferencing the base iterator.
    template <typename Iterator, typename Transformer, typename Reference,
        typename Value>
    class transform_iterator
      : public detail::transform_iterator_base<
            Iterator, Transformer, Reference, Value
        >::type
    {
    private:
        typedef typename detail::transform_iterator_base<
                Iterator, Transformer, Reference, Value
            >::type base_type;

    public:
        transform_iterator() {}

        explicit transform_iterator(Iterator const& it)
          : base_type(it)
        {}
        transform_iterator(Iterator const& it, Transformer const& f)
          : base_type(it), transformer_(f)
        {}

        Transformer const& transformer() const
        {
            return transformer_;
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & transformer_;
        }

    private:
        friend class boost::iterator_core_access;

        typename base_type::reference dereference() const
        {
            return transformer_(this->base());
        }

        Transformer transformer_;
    };
}}

// namespace hpx { namespace traits
// {
//     template <typename Iterator, typename Transformer, typename Reference,
//         typename Value>
//     struct segmented_iterator_traits<
//         util::transform_iterator<Iterator, Transformer, Reference, Value>,
//         typename boost::enable_if<
//             typename segmented_iterator_traits<Iterator>::is_segmented_iterator
//         >::type>
//     {
//         typedef std::true_type is_segmented_iterator;
//         typedef segmented_iterator_traits<Iterator> base_traits;
//         typedef traits::transform_iterator_transformer_traits<Transformer>
//             transformer_traits;
//
//         typedef util::transform_iterator<Iterator, Transformer> iterator;
//
//         typedef util::transform_iterator<
//                 typename base_traits::segment_iterator,
//                 typename util::result_of<transformer_traits(
//                     Transformer, traits::functional::segmented_iterator_segment
//                 )>::type,
//                 Reference, Value
//             > segment_iterator;
//         typedef util::transform_iterator<
//                 typename base_traits::local_iterator,
//                 typename util::result_of<transformer_traits(
//                     Transformer, traits::functional::segmented_iterator_local
//                 )>::type,
//                 Reference, Value
//             > local_iterator;
//
//         typedef util::transform_iterator<
//                 typename base_traits::local_segment_iterator, Transformer,
//                 Reference, Value
//             > local_segment_iterator;
//         typedef util::transform_iterator<
//                 typename base_traits::local_raw_iterator, Transformer,
//                 Reference, Value
//             > local_raw_iterator;
//
//         //  Conceptually this function is supposed to denote which segment
//         //  the iterator is currently pointing to (i.e. just global iterator).
//         static segment_iterator segment(iterator iter)
//         {
//             return segment_iterator(
//                 base_traits::segment(iter.base()),
//                 transformer_traits()(
//                     iter.transformer(),
//                     traits::functional::segmented_iterator_segment()
//                 ));
//         }
//
//         //  This function should specify which is the current segment and
//         //  the exact position to which local iterator is pointing.
//         static local_iterator local(iterator iter)
//         {
//             return local_iterator(
//                 base_traits::local(iter.base()),
//                 transformer_traits()(
//                     iter.transformer(),
//                     traits::functional::segmented_iterator_local()
//                 ));
//         }
//
//         //  This function should specify the local iterator which is at the
//         //  beginning of the partition.
//         static local_iterator begin(segment_iterator const& iter)
//         {
//             return local_iterator(
//                 base_traits::begin(iter.base()),
//                 transformer_traits()(
//                     iter.transformer(),
//                     traits::functional::segmented_iterator_begin()
//                 ));
//         }
//
//         //  This function should specify the local iterator which is at the
//         //  end of the partition.
//         static local_iterator end(segment_iterator const& iter)
//         {
//             return local_iterator(
//                 base_traits::end(iter.base()),
//                 transformer_traits()(
//                     iter.transformer(),
//                     traits::functional::segmented_iterator_end()
//                 ));
//         }
//
//         //  This function should specify the local iterator which is at the
//         //  beginning of the partition data.
//         static local_raw_iterator begin(local_segment_iterator const& seg_iter)
//         {
//             return local_raw_iterator(
//                 base_traits::begin(seg_iter.base()),
//                 transformer_traits()(
//                     iter.transformer(),
//                     traits::functional::segmented_iterator_local_begin()
//                 ));
//         }
//
//         //  This function should specify the local iterator which is at the
//         //  end of the partition data.
//         static local_raw_iterator end(local_segment_iterator const& seg_iter)
//         {
//             return local_raw_iterator(
//                 base_traits::end(seg_iter.base()),
//                 transformer_traits()(
//                     iter.transformer(),
//                     traits::functional::segmented_iterator_local_end()
//                 ));
//         }
//
//         // Extract the base id for the segment referenced by the given segment
//         // iterator.
//         static id_type get_id(segment_iterator const& iter)
//         {
//             return base_traits::get_id(iter.base());
//         }
//     };
//
//     template <typename Iterator, typename Transformer, typename Reference,
//         typename Value>
//     struct segmented_local_iterator_traits<
//         util::transform_iterator<Iterator, Transformer, Reference, Value>,
//         typename boost::enable_if<
//             typename segmented_local_iterator_traits<
//                 Iterator
//             >::is_segmented_local_iterator
//         >::type>
//     {
//         typedef std::true_type is_segmented_local_iterator;
//         typedef segmented_local_iterator_traits<Iterator> base_traits;
//
//         typedef util::transform_iterator<
//                 typename base_traits::iterator, Transformer,
//                 Reference, Value
//             > iterator;
//         typedef util::transform_iterator<
//                 typename base_traits::local_iterator, Transformer,
//                 Reference, Value
//             > local_iterator;
//         typedef util::transform_iterator<
//                 typename base_traits::local_raw_iterator, Transformer,
//                 Reference, Value
//             > local_raw_iterator;
//
//         // Extract base iterator from local_iterator
//         static local_raw_iterator local(local_iterator const& iter)
//         {
//             return local_raw_iterator(base_traits::local(iter.base()));
//         }
//
//         // Construct remote local_iterator from local_raw_iterator
//         static local_iterator remote(local_raw_iterator const& iter)
//         {
//             return local_iterator(base_traits::remote(iter.base()));
//         }
//     };
// }}

#endif

