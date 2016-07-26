//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http:// ww.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UNORDERED_MAP_SEGMENTED_ITERATOR_HPP
#define HPX_UNORDERED_MAP_SEGMENTED_ITERATOR_HPP

/// \file hpx/components/unordered_map/unordered_map_segmented_iterator.hpp
/// \brief This file contains the implementation of iterators for hpx::unordered_map.

 // The idea for these iterators is taken from
 // http://lafstern.org/matt/segmented.pdf.

#include <hpx/config.hpp>
//#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/iterator_adaptor.hpp>

#include <hpx/components/containers/unordered/partition_unordered_map_component.hpp>

#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <boost/integer.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Key, typename T, typename Hash = std::hash<Key>,
        typename KeyEqual = std::equal_to<Key> >
    class unordered_map;

    template <typename Key, typename T, typename Hash, typename KeyEqual,
        typename BaseIter>
    class segment_unordered_map_iterator;
    template <typename Key, typename T, typename Hash, typename KeyEqual,
        typename BaseIter>
    class const_segment_unordered_map_iterator;

    ///////////////////////////////////////////////////////////////////////////
    // This class wraps plain a vector<>::iterator or vector<>::const_iterator

    /// This class implement the segmented iterator for the hpx::vector
    template <typename Key, typename T, typename Hash, typename KeyEqual,
        typename BaseIter>
    class segment_unordered_map_iterator
      : public hpx::util::iterator_adaptor<
            segment_unordered_map_iterator<Key, T, Hash, KeyEqual, BaseIter>,
            BaseIter
        >
    {
    private:
        typedef hpx::util::iterator_adaptor<
                segment_unordered_map_iterator<Key, T, Hash, KeyEqual, BaseIter>,
                BaseIter
            > base_type;

    public:
        segment_unordered_map_iterator(BaseIter const& it,
                unordered_map<Key, T, Hash, KeyEqual>* data = nullptr)
          : base_type(it), data_(data)
        {}

        unordered_map<Key, T, Hash, KeyEqual>* get_data()
        {
            return data_;
        }
        unordered_map<Key, T, Hash, KeyEqual> const* get_data() const
        {
            return data_;
        }

        bool is_at_end() const
        {
            return data_ == 0 ||
                this->base_type::base_reference() == data_->partitions_.end();
        }

    private:
        unordered_map<Key, T, Hash, KeyEqual>* data_;
    };

    template <typename Key, typename T, typename Hash, typename KeyEqual,
        typename BaseIter>
    class const_segment_unordered_map_iterator
      : public hpx::util::iterator_adaptor<
            const_segment_unordered_map_iterator<
                Key, T, Hash, KeyEqual, BaseIter>,
            BaseIter
        >
    {
    private:
        typedef hpx::util::iterator_adaptor<
                const_segment_unordered_map_iterator<
                    Key, T, Hash, KeyEqual, BaseIter>,
                BaseIter
            > base_type;

    public:
        const_segment_unordered_map_iterator(BaseIter const& it,
                unordered_map<Key, T, Hash, KeyEqual> const* data = nullptr)
          : base_type(it), data_(data)
        {}

        unordered_map<Key, T, Hash, KeyEqual> const* get_data() const
        {
            return data_;
        }

        bool is_at_end() const
        {
            return data_ == 0 ||
                this->base_type::base_reference() == data_->partitions_.end();
        }

    private:
        unordered_map<Key, T, Hash, KeyEqual> const* data_;
    };

//     ///////////////////////////////////////////////////////////////////////////
//     namespace detail
//     {
//         template <typename BaseIterator>
//         struct is_requested_locality
//         {
//             typedef typename std::iterator_traits<BaseIterator>::reference
//                 reference;
//
//             is_requested_locality(boost::uint32_t locality_id =
//                     naming::invalid_locality_id)
//               : locality_id_(locality_id)
//             {}
//
//             bool operator()(reference val) const
//             {
//                 return locality_id_ == naming::invalid_locality_id ||
//                        locality_id_ == val.locality_id_;
//             }
//
//             boost::uint32_t locality_id_;
//         };
//     }
//
//     /// This class implement the local segmented iterator for the hpx::vector
//     template <typename T, typename BaseIter>
//     class local_segment_vector_iterator
//       : public hpx::util::iterator_adaptor<
//             local_segment_vector_iterator<T, BaseIter>, BaseIter,
//             std::vector<T>, std::forward_iterator_tag
//         >
//     {
//     private:
//         typedef hpx::util::iterator_adaptor<
//                 local_segment_vector_iterator<T, BaseIter>, BaseIter,
//                 std::vector<T>, std::forward_iterator_tag
//             > base_type;
//         typedef detail::is_requested_locality<BaseIter> predicate;
//
//     public:
//         local_segment_vector_iterator(BaseIter const& end)
//           : base_type(end), predicate_(), end_(end)
//         {}
//
//         local_segment_vector_iterator(
//                 BaseIter const& it, BaseIter const& end,
//                 boost::uint32_t locality_id)
//           : base_type(it), predicate_(locality_id), end_(end)
//         {
//             satisfy_predicate();
//         }
//
//         bool is_at_end() const
//         {
//             return !data_ || this->base() == end_;
//         }
//
//     private:
//         friend class hpx::util::iterator_core_access;
//
//         typename base_type::reference dereference() const
//         {
//             HPX_ASSERT(!is_at_end());
//             return data_->get_data();
//         }
//
//         void increment()
//         {
//             ++(this->base_reference());
//             satisfy_predicate();
//         }
//
//         void satisfy_predicate()
//         {
//             while (this->base() != end_ && !predicate_(*this->base()))
//                 ++(this->base_reference());
//
//             if (this->base() != end_)
//                 data_ = this->base()->local_data_;
//             else
//                 data_.reset();
//         }
//
//     private:
//         std::shared_ptr<server::partitioned_vector<T> > data_;
//         predicate predicate_;
//         BaseIter end_;
//     };
}

#endif
