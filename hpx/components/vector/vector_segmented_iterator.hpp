//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http:// ww.boost.org/LICENSE_1_0.txt)

#ifndef HPX_VECTOR_SEGMENTED_ITERATOR_HPP
#define HPX_VECTOR_SEGMENTED_ITERATOR_HPP

/// \file hpx/components/vector/segmented_iterator.hpp
/// \brief This file contains the implementation of iterators for hpx::vector.

 // The idea for these iterators is taken from
 // http://afstern.org/matt/segmented.pdf.

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/util.hpp>

#include <hpx/components/vector/partition_vector_component.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <cstdint>
#include <iterator>

#include <boost/integer.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/filter_iterator.hpp>

#include <boost/serialization/serialization.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T> class vector;       // forward declaration

    template <typename T> class local_vector_iterator;
    template <typename T> class const_local_vector_iterator;

    template <typename T> class vector_iterator;
    template <typename T> class const_vector_iterator;

    template <typename T, typename BaseIter> class segment_vector_iterator;
    template <typename T, typename BaseIter> class const_segment_vector_iterator;

    namespace server
    {
        template <typename T> class partition_vector;
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct local_vector_value_proxy
        {
            local_vector_value_proxy(local_vector_iterator<T> const& it)
              : it_(it)
            {}

            operator T() const
            {
                if (!it_.get_data())
                {
                    return it_.get_partition().get_value_sync(
                        it_.get_local_index());
                }
                return *(it_.get_data()->begin() + it_.get_local_index());
            }

            template <typename T_>
            local_vector_value_proxy& operator=(T_ && value)
            {
                if (!it_.get_data())
                {
                    it_.get_partition().set_value_sync(
                        it_.get_local_index(), std::forward<T_>(value));
                }
                else
                {
                    *(it_.get_data()->begin() + it_.get_local_index()) =
                        std::forward<T_>(value);
                }
                return *this;
            }

            local_vector_iterator<T> const& it_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct vector_value_proxy
        {
            vector_value_proxy(hpx::vector<T>& v, std::size_t index)
              : v_(v), index_(index)
            {}

            operator T() const
            {
                return v_.get_value_sync(index_);
            }

            template <typename T_>
            vector_value_proxy& operator=(T_ && value)
            {
                v_.set_value_sync(index_, std::forward<T_>(value));
                return *this;
            }

            vector<T>& v_;
            std::size_t index_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This class implements the local iterator functionality for
    /// the partitioned backend of a hpx::vector.
    template <typename T>
    class local_vector_iterator
      : public boost::iterator_facade<
            local_vector_iterator<T>, T, std::random_access_iterator_tag,
            detail::local_vector_value_proxy<T>
        >
    {
    private:
        typedef boost::iterator_facade<
                local_vector_iterator<T>, T, std::random_access_iterator_tag,
                detail::local_vector_value_proxy<T>
            > base_type;

    public:
        typedef std::size_t size_type;

        // constructors
        local_vector_iterator()
          : partition_(), local_index_(size_type(-1))
        {}

        local_vector_iterator(partition_vector<T> partition,
                size_type local_index)
          : partition_(partition),
            local_index_(local_index)
        {}

        local_vector_iterator(partition_vector<T> partition,
                size_type local_index,
                boost::shared_ptr<server::partition_vector<T> > const& data)
          : partition_(partition),
            local_index_(local_index),
            data_(data)
        {}

        typedef typename std::vector<T>::iterator base_iterator_type;
        typedef typename std::vector<T>::const_iterator base_const_iterator_type;

        ///////////////////////////////////////////////////////////////////////
        base_iterator_type base_iterator()
        {
            HPX_ASSERT(data_);
            return data_->begin() + local_index_;
        }
        base_const_iterator_type base_iterator() const
        {
            HPX_ASSERT(data_);
            return data_->cbegin() + local_index_;
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void load(Archive& ar, unsigned version)
        {
            ar & partition_ & local_index_;
            if (partition_)
                data_ = partition_.get_ptr();
        }
        template <typename Archive>
        void save(Archive& ar, unsigned version) const
        {
            ar & partition_ & local_index_;
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

        bool is_at_end() const
        {
            return !partition_ || local_index_ == size_type(-1);
        }

    protected:
        friend class boost::iterator_core_access;

        bool equal(local_vector_iterator const& other) const
        {
            if (is_at_end())
                return other.is_at_end();

            return partition_ == other.partition_ &&
                local_index_ == other.local_index_;
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(!is_at_end());
            return detail::local_vector_value_proxy<T>(*this);
        }

        void increment()
        {
            HPX_ASSERT(!is_at_end());
            ++local_index_;
        }

        void decrement()
        {
            HPX_ASSERT(!is_at_end());
            --local_index_;
        }

        void advance(std::ptrdiff_t n)
        {
            HPX_ASSERT(!is_at_end());
            local_index_ += n;
        }

        std::ptrdiff_t distance_to(local_vector_iterator const& other) const
        {
            if (other.is_at_end())
            {
                if (is_at_end())
                    return 0;
                if (data_)
                    return data_->size() - local_index_;
                return partition_.size() - local_index_;
            }

            if (is_at_end())
            {
                if (!other.data_)
                    return other.local_index_ - other.partition_.size();
                return other.local_index_ - other.data_->size();
            }

            HPX_ASSERT(partition_ == other.partition_);
            return other.local_index_ - local_index_;
        }

    public:
        partition_vector<T>& get_partition() { return partition_; }
        partition_vector<T> get_partition() const { return partition_; }

        size_type get_local_index() const { return local_index_; }

        boost::shared_ptr<server::partition_vector<T> >& get_data()
        {
            return data_;
        }
        boost::shared_ptr<server::partition_vector<T> > const& get_data() const
        {
            return data_;
        }

    protected:
        // refer to a partition of the vector
        partition_vector<T> partition_;

        // local position in the referenced partition
        size_type local_index_;

        // caching address of component
        boost::shared_ptr<server::partition_vector<T> > data_;
    };

    template <typename T>
    class const_local_vector_iterator
      : public boost::iterator_facade<
            const_local_vector_iterator<T>, T const,
            std::random_access_iterator_tag, T const
        >
    {
    private:
        typedef boost::iterator_facade<
                const_local_vector_iterator<T>, T const,
                std::random_access_iterator_tag, T const
            > base_type;

    public:
        typedef std::size_t size_type;

        // constructors
        const_local_vector_iterator()
          : partition_(), local_index_(size_type(-1))
        {}

        const_local_vector_iterator(partition_vector<T> partition,
                size_type local_index)
          : partition_(partition),
            local_index_(local_index)
        {}

        const_local_vector_iterator(partition_vector<T> partition,
                size_type local_index,
                boost::shared_ptr<server::partition_vector<T> > const& data)
          : partition_(partition),
            local_index_(local_index),
            data_(data)
        {}

        typedef typename std::vector<T>::const_iterator base_iterator_type;
        typedef typename std::vector<T>::const_iterator base_const_iterator_type;

        ///////////////////////////////////////////////////////////////////////
        base_const_iterator_type base_iterator()
        {
            HPX_ASSERT(data_);
            return data_->cbegin() + local_index_;
        }
        base_const_iterator_type base_iterator() const
        {
            HPX_ASSERT(data_);
            return data_->cbegin() + local_index_;
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void load(Archive& ar, unsigned version)
        {
            ar & partition_ & local_index_;
            if (partition_)
                data_ = partition_.get_ptr();
        }
        template <typename Archive>
        void save(Archive& ar, unsigned version) const
        {
            ar & partition_ & local_index_;
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

        bool is_at_end() const
        {
            return !partition_ || local_index_ == size_type(-1);
        }

    protected:
        friend class boost::iterator_core_access;

        bool equal(const_local_vector_iterator const& other) const
        {
            if (is_at_end())
                return other.is_at_end();

            return partition_ == other.partition_ &&
                local_index_ == other.local_index_;
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(!is_at_end());
            return *base_iterator();
        }

        void increment()
        {
            HPX_ASSERT(!is_at_end());
            ++local_index_;
        }

        void decrement()
        {
            HPX_ASSERT(!is_at_end());
            --local_index_;
        }

        void advance(std::ptrdiff_t n)
        {
            HPX_ASSERT(!is_at_end());
            local_index_ += n;
        }

        std::ptrdiff_t distance_to(const_local_vector_iterator const& other) const
        {
            if (other.is_at_end())
            {
                if (is_at_end())
                    return 0;
                if (data_)
                    return data_->size() - local_index_;
                return partition_.size() - local_index_;
            }

            if (is_at_end())
            {
                if (!other.data_)
                    return other.local_index_ - other.partition_.size();
                return other.local_index_ - other.data_->size();
            }

            HPX_ASSERT(partition_ == other.partition_);
            return other.local_index_ - local_index_;
        }

    public:
        partition_vector<T> const& get_partition() const { return partition_; }
        size_type get_local_index() const { return local_index_; }

    protected:
        // refer to a partition of the vector
        partition_vector<T> partition_;

        // local position in the referenced partition
        size_type local_index_;

        // caching address of component
        boost::shared_ptr<server::partition_vector<T> > data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseIterator>
    struct is_requested_locality
    {
        typedef typename std::iterator_traits<BaseIterator>::reference reference;

        is_requested_locality(boost::uint32_t locality_id = naming::invalid_locality_id)
          : locality_id_(locality_id)
        {}

        bool operator()(reference val) const
        {
            return locality_id_ == naming::invalid_locality_id ||
                   locality_id_ == val.locality_id_;
        }

        boost::uint32_t locality_id_;
    };

    /// \brief This class implement the segmented iterator for the hpx::vector
    template <typename T, typename BaseIter>
    class segment_vector_iterator
      : public boost::filter_iterator<
            is_requested_locality<BaseIter>, BaseIter
        >
    {
    private:
        typedef boost::filter_iterator<
            is_requested_locality<BaseIter>, BaseIter
        > base_type;
        typedef is_requested_locality<BaseIter> predicate;

    public:
        segment_vector_iterator()
          : data_(0)
        {}

        segment_vector_iterator(vector<T>* data, BaseIter const& end)
          : base_type(predicate(), end, end), data_(data)
        {}

        segment_vector_iterator(
                vector<T>* data, BaseIter const& it, BaseIter const& end,
                boost::uint32_t locality_id = naming::invalid_locality_id)
          : base_type(predicate(locality_id), it, end), data_(data)
        {}

        vector<T>* get_data() { return data_; }
        vector<T> const* get_data() const { return data_; }

        bool is_at_end() const
        {
            return data_ == 0 || this->base() == this->end();
        }

    private:
        vector<T>* data_;
    };

    template <typename T, typename BaseIter>
    class const_segment_vector_iterator
      : public boost::filter_iterator<
            is_requested_locality<BaseIter>, BaseIter
        >
    {
    private:
        typedef boost::filter_iterator<
            is_requested_locality<BaseIter>, BaseIter
        > base_type;
        typedef is_requested_locality<BaseIter> predicate;

    public:
        const_segment_vector_iterator()
          : data_(0)
        {}

        const_segment_vector_iterator(vector<T> const* data, BaseIter const& end)
          : base_type(predicate(), end, end), data_(data)
        {}

        const_segment_vector_iterator(
                vector<T> const* data, BaseIter const& it, BaseIter const& end,
                boost::uint32_t locality_id = naming::invalid_locality_id)
          : base_type(predicate(locality_id), it, end), data_(data)
        {}

        vector<T> const* get_data() const { return data_; }

        bool is_at_end() const
        {
            return data_ == 0 || this->base() == this->end();
        }

    private:
        vector<T> const* data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This class implements the iterator functionality for hpx::vector.
    template <typename T>
    class vector_iterator
      : public boost::iterator_facade<
            vector_iterator<T>, T, std::random_access_iterator_tag,
            detail::vector_value_proxy<T>
        >
    {
    private:
        typedef boost::iterator_facade<
                vector_iterator<T>, T, std::random_access_iterator_tag,
                detail::vector_value_proxy<T>
            > base_type;

    public:
        typedef std::size_t size_type;
        typedef typename vector<T>::segment_iterator segment_iterator;
        typedef typename vector<T>::local_iterator local_iterator;

        // constructors
        vector_iterator()
          : data_(0), global_index_(size_type(-1))
        {}

        vector_iterator(vector<T>* data, size_type global_index)
          : data_(data), global_index_(global_index)
        {}

        vector<T>* get_data() { return data_; }
        vector<T> const* get_data() const { return data_; }

        size_type get_global_index() const { return global_index_; }

        bool is_at_end() const
        {
            return data_ == 0 || global_index_ == size_type(-1);
        }

    protected:
        friend class boost::iterator_core_access;

        bool equal(vector_iterator const& other) const
        {
            if (is_at_end())
                return other.is_at_end();
            return data_ == other.data_ && global_index_ == other.global_index_;
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(!is_at_end());
            return detail::vector_value_proxy<T>(*data_, global_index_);
        }

        void increment()
        {
            HPX_ASSERT(!is_at_end());
            ++global_index_;
        }

        void decrement()
        {
            HPX_ASSERT(!is_at_end());
            --global_index_;
        }

        void advance(std::ptrdiff_t n)
        {
            HPX_ASSERT(!is_at_end());
            global_index_ += n;
        }

        std::ptrdiff_t distance_to(vector_iterator const& other) const
        {
            if (other.is_at_end())
            {
                if (is_at_end())
                    return 0;
                return data_->size() - global_index_;
            }

            if(is_at_end())
                return other.global_index_ - other.data_->size();

            HPX_ASSERT(data_ == other.data_);
            return other.global_index_ - global_index_;
        }

    protected:
        // refer to the vector
        vector<T>* data_;

        // global position in the referenced vector
        size_type global_index_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class const_vector_iterator
      : public boost::iterator_facade<
            const_vector_iterator<T>, T const,
            std::random_access_iterator_tag, T const
        >
    {
    private:
        typedef boost::iterator_facade<
                const_vector_iterator<T>, T const,
                std::random_access_iterator_tag, T const
            > base_type;

    public:
        typedef std::size_t size_type;
        typedef typename vector<T>::const_segment_iterator segment_iterator;
        typedef typename vector<T>::const_local_iterator local_iterator;

        // constructors
        const_vector_iterator()
          : data_(0), global_index_(size_type(-1))
        {}

        const_vector_iterator(vector<T> const* data, size_type global_index)
          : data_(data), global_index_(global_index)
        {}

        vector<T> const* get_data() const { return data_; }
        size_type get_global_index() const { return global_index_; }

        bool is_at_end() const
        {
            return data_ == 0 || global_index_ == size_type(-1);
        }

    protected:
        friend class boost::iterator_core_access;

        bool equal(const_vector_iterator const& other) const
        {
            if (is_at_end())
                return other.is_at_end();
            return data_ == other.data_ && global_index_ == other.global_index_;
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(!is_at_end());
            return data_->get_value_sync(global_index_);
        }

        void increment()
        {
            HPX_ASSERT(!is_at_end());
            ++global_index_;
        }

        void decrement()
        {
            HPX_ASSERT(!is_at_end());
            --global_index_;
        }

        void advance(std::ptrdiff_t n)
        {
            HPX_ASSERT(!is_at_end());
            global_index_ += n;
        }

        std::ptrdiff_t distance_to(const_vector_iterator const& other) const
        {
            if (other.is_at_end())
            {
                if (is_at_end())
                    return 0;
                return data_->size() - global_index_;
            }

            if(is_at_end())
                return other.global_index_ - other.data_->size();

            HPX_ASSERT(data_ == other.data_);
            return other.global_index_ - global_index_;
        }

    protected:
        // refer to the vector
        vector<T> const* data_;

        // global position in the referenced vector
        size_type global_index_;
    };
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename T>
    struct segmented_iterator_traits<vector_iterator<T> >
    {
        typedef boost::mpl::true_ is_segmented_iterator;

        typedef vector_iterator<T> iterator;
        typedef typename iterator::segment_iterator segment_iterator;
        typedef typename iterator::local_iterator local_iterator;

        //  Conceptually this function is supposed to denote which segment
        //  the iterator is currently pointing to (i.e. just global iterator).
        static segment_iterator segment(iterator iter)
        {
            return iter.get_data()->get_segment_iterator(
                iter.get_global_index());
        }

        //  This function should specify which is the current segment and
        //  the exact position to which local iterator is pointing.
        static local_iterator local(iterator iter)
        {
            if (iter.is_at_end())           // avoid dereferencing end iterator
                return local_iterator();

            return iter.get_data()->get_local_iterator(
                iter.get_global_index());
        }

        //  Build a full iterator from the segment and local iterators
        static iterator compose(segment_iterator seg_iter,
            local_iterator& local_iter)
        {
            vector<T>* data = seg_iter.get_data();
            std::size_t index = local_iter.get_local_index();
            return iterator(data, data->get_global_index(seg_iter, index));
        }

        //  This function should specify the local iterator which is at the
        //  beginning of the partition.
        static local_iterator begin(segment_iterator const& seg_iter)
        {
            if (seg_iter.is_at_end())       // avoid dereferencing end iterator
                return local_iterator();

            return local_iterator(seg_iter.base()->partition_, 0,
                seg_iter.base()->local_data_);
        }

        //  This function should specify the local iterator which is at the
        //  end of the partition.
        static local_iterator end(segment_iterator const& seg_iter)
        {
            if (seg_iter.is_at_end())       // avoid dereferencing end iterator
                return local_iterator();

            return local_iterator(seg_iter.base()->partition_,
                seg_iter.base()->size_);
        }
    };

    template <typename T>
    struct segmented_iterator_traits<const_vector_iterator<T> >
    {
        typedef boost::mpl::true_ is_segmented_iterator;

        typedef const_vector_iterator<T> iterator;
        typedef typename iterator::segment_iterator segment_iterator;
        typedef typename iterator::local_iterator local_iterator;

        //  Conceptually this function is supposed to denote which segment
        //  the iterator is currently pointing to (i.e. just global iterator).
        static segment_iterator segment(iterator iter)
        {
            return iter.get_data()->get_const_segment_iterator(
                iter.get_global_index());
        }

        //  This function should specify which is the current segment and
        //  the exact position to which local iterator is pointing.
        static local_iterator local(iterator const& iter)
        {
            if (iter.is_at_end())           // avoid dereferencing end iterator
                return local_iterator();

            return iter.get_data()->get_const_local_iterator(
                iter.get_global_index());
        }

        //  Build a full iterator from the segment and local iterators
        static iterator compose(segment_iterator const& seg_iter,
            local_iterator const& local_iter)
        {
            vector<T> const* data = seg_iter.get_data();
            std::size_t index = local_iter.get_local_index();
            return iterator(data, data->get_global_index(seg_iter, index));
        }

        //  This function should specify the local iterator which is at the
        //  beginning of the partition.
        static local_iterator begin(segment_iterator const& seg_iter)
        {
            if (seg_iter.is_at_end())       // avoid dereferencing end iterator
                return local_iterator();

            return local_iterator(seg_iter.base()->partition_, 0,
                seg_iter.base()->local_data_);
        }

        //  This function should specify the local iterator which is at the
        //  end of the partition.
        static local_iterator end(segment_iterator const& seg_iter)
        {
            if (seg_iter.is_at_end())       // avoid dereferencing end iterator
                return local_iterator();

            return local_iterator(seg_iter.base()->partition_,
                seg_iter.base()->size_);
        }
    };
}}

#endif //  SEGMENTED_ITERATOR_HPP
