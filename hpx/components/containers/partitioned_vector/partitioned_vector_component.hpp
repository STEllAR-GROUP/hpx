//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/partitioned_vector/partitioned_vector_component.hpp

#ifndef HPX_PARTITIONED_VECTOR_COMPONENT_HPP
#define HPX_PARTITIONED_VECTOR_COMPONENT_HPP

/// \brief The partitioned_vector_partition as the hpx component is defined here.
///
/// The partitioned_vector_partition is the wrapper to the stl vector class except all API's
/// are defined as component action. All the API's in stubs classes are
/// asynchronous API which return the futures.

#include <hpx/config.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/runtime/components/server/component.hpp>
#include <hpx/runtime/get_ptr.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/util/assert.hpp>

#include <hpx/components/containers/partitioned_vector/partitioned_vector_fwd.hpp>

#include <boost/preprocessor/cat.hpp>

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace hpx { namespace server
{
    /// \brief This is the basic wrapper class for stl vector.
    ///
    /// This contain the implementation of the partitioned_vector_partition's
    /// component functionality.
    template <typename T, typename Data>
    class partitioned_vector
      : public components::locking_hook<
            components::component_base<partitioned_vector<T, Data> >
        >
    {
    public:
        typedef Data data_type;

        typedef typename data_type::allocator_type allocator_type;
        typedef typename data_type::size_type size_type;
        typedef typename data_type::iterator iterator_type;
        typedef typename data_type::const_iterator const_iterator_type;

        typedef components::locking_hook<
                components::component_base<partitioned_vector<T, Data> > >
            base_type;

        data_type partitioned_vector_partition_;

        ///////////////////////////////////////////////////////////////////////
        // Constructors
        ///////////////////////////////////////////////////////////////////////

        /// Default Constructor which create partitioned_vector_partition with
        /// size 0.
        partitioned_vector()
        {
            HPX_ASSERT(false);  // shouldn't ever be called
        }

        explicit partitioned_vector(size_type partition_size)
          : partitioned_vector_partition_(partition_size)
        {}

        /// Constructor which create and initialize partitioned_vector_partition
        /// with all elements as \a val.
        ///
        /// param partition_size The size of vector
        /// param val Default value for the elements in partitioned_vector_partition
        ///
        partitioned_vector(size_type partition_size, T const& val)
          : partitioned_vector_partition_(partition_size, val)
        {}

        partitioned_vector(size_type partition_size, T const& val,
                allocator_type const& alloc)
          : partitioned_vector_partition_(partition_size, val, alloc)
        {}

        // support components::copy
        partitioned_vector(partitioned_vector const& rhs)
          : base_type(rhs),
            partitioned_vector_partition_(rhs.partitioned_vector_partition_)
        {}

        partitioned_vector& operator=(partitioned_vector const& rhs)
        {
            if (this != &rhs)
            {
                this->base_type::operator=(rhs);
                partitioned_vector_partition_ = rhs.partitioned_vector_partition_;
            }
            return *this;
        }

        partitioned_vector(partitioned_vector && rhs)
          : base_type(std::move(rhs)),
            partitioned_vector_partition_(std::move(rhs.partitioned_vector_partition_))
        {}

        partitioned_vector& operator=(partitioned_vector && rhs)
        {
            if (this != &rhs)
            {
                this->base_type::operator=(std::move(rhs));
                partitioned_vector_partition_ =
                    std::move(rhs.partitioned_vector_partition_);
            }
            return *this;
        }

        ///////////////////////////////////////////////////////////////////////
        data_type& get_data()
        {
            return partitioned_vector_partition_;
        }
        data_type const& get_data() const
        {
            return partitioned_vector_partition_;
        }

        /// Duplicate the copy method for action naming
        data_type get_copied_data() const
        {
            return partitioned_vector_partition_;
        }

        ///////////////////////////////////////////////////////////////////////
        void set_data(data_type && other)
        {
            partitioned_vector_partition_ = std::move(other);
        }

        ///////////////////////////////////////////////////////////////////////
        iterator_type begin()
        {
            return partitioned_vector_partition_.begin();
        }
        const_iterator_type begin() const
        {
            return partitioned_vector_partition_.begin();
        }
        const_iterator_type cbegin() const
        {
            return partitioned_vector_partition_.cbegin();
        }

        iterator_type end()
        {
            return partitioned_vector_partition_.end();
        }
        const_iterator_type end() const
        {
            return partitioned_vector_partition_.end();
        }
        const_iterator_type cend() const
        {
            return partitioned_vector_partition_.cend();
        }

        ///////////////////////////////////////////////////////////////////////
        // Capacity Related API's in the server class
        ///////////////////////////////////////////////////////////////////////

        /// Returns the number of elements
        size_type size() const
        {
            return partitioned_vector_partition_.size();
        }

        /// Returns the maximum possible number of elements
        size_type max_size() const
        {
            return partitioned_vector_partition_.max_size();
        }

        /// Returns the number of elements that the container has currently
        /// allocated space for.
        size_type capacity() const
        {
            return partitioned_vector_partition_.capacity();
        }

        /// Checks if the container has no elements, i.e. whether
        /// begin() == end().
        bool empty() const
        {
            return partitioned_vector_partition_.empty();
        }

        /// Changes the number of elements stored .
        ///
        /// \param n    new size of the partitioned_vector_partition
        /// \param val  value to be copied if \a n is greater than the
        ///              current size
        ///
        void resize(size_type n, T const& val)
        {
            partitioned_vector_partition_.resize(n, val);
        }

        /// Request the change in partitioned_vector_partition capacity so that it
        /// can hold \a n elements.
        ///
        /// This function request partitioned_vector_partition capacity should
        /// be at least enough to contain n elements. If n is greater than current
        /// partitioned_vector_partition capacity, the function causes the
        /// partitioned_vector_partition to
        /// reallocate its storage increasing its capacity to n (or greater).
        /// In other cases the partitioned_vector_partition capacity does not
        /// got affected.
        /// It does not change the partitioned_vector_partition size.
        ///
        /// \param n minimum capacity of partitioned_vector_partition
        ///
        ///
        void reserve(size_type n)
        {
            partitioned_vector_partition_.reserve(n);
        }

        ///////////////////////////////////////////////////////////////////////
        // Element access API's
        ///////////////////////////////////////////////////////////////////////

        /// Return the element at the position \a pos in the
        /// partitioned_vector_partition container.
        ///
        /// \param pos Position of the element in the partitioned_vector_partition
        ///
        /// \return Return the value of the element at position represented
        ///         by \a pos.
        ///
        T get_value(size_type pos) const
        {
            return partitioned_vector_partition_[pos];
        }

        /// Return the element at the position \a pos in the
        /// partitioned_vector_partition container.
        ///
        /// \param pos Positions of the elements in the
        ///            partitioned_vector_partition
        ///
        /// \return Return the values of the elements at position represented
        ///         by \a pos.
        ///
        std::vector<T> get_values(std::vector<size_type> const& pos) const
        {
            std::vector<T> result;
            result.reserve(pos.size());

            for (std::size_t i = 0; i != pos.size(); ++i)
                result.push_back(partitioned_vector_partition_[pos[i]]);

            return result;
        }


        /// Access the value of first element in the partitioned_vector_partition.
        ///
        /// Calling the function on empty container cause undefined behavior.
        ///
        /// \return Return the value of the first element in the
        ///         partitioned_vector_partition
        ///
        T front() const
        {
            return partitioned_vector_partition_.front();
        }

        /// Access the value of last element in the partitioned_vector_partition.
        ///
        /// Calling the function on empty container cause undefined behavior.
        ///
        /// \return Return the value of the last element in the
        ///         partitioned_vector_partition
        ///
        T back() const
        {
            return partitioned_vector_partition_.back();
        }

        ///////////////////////////////////////////////////////////////////////
        // Modifiers API's in server class
        ///////////////////////////////////////////////////////////////////////

        /// Assigns new contents to the partitioned_vector_partition, replacing
        /// its current contents and modifying its size accordingly.
        ///
        /// \param n     new size of partitioned_vector_partition
        /// \param val   Value to fill the container with
        ///
        void assign(size_type n, T const& val)
        {
            partitioned_vector_partition_.assign(n, val);
        }

        /// Add new element at the end of partitioned_vector_partition. The added
        /// element contain the \a val as value.
        ///
        /// \param val Value to be copied to new element
        ///
        void push_back(T const& val)
        {
            partitioned_vector_partition_.push_back(val);
        }

        /// Remove the last element from partitioned_vector_partition effectively
        /// reducing the size by one. The removed element is destroyed.
        ///
        void pop_back()
        {
            partitioned_vector_partition_.pop_back();
        }

        //  This API is required as we do not returning the reference to the
        //  element in Any API.

        /// Copy the value of \a val in the element at position \a pos in the
        /// partitioned_vector_partition container.
        ///
        /// \param pos   Position of the element in the
        ///              partitioned_vector_partition
        ///
        /// \param val   The value to be copied
        ///
        void set_value(size_type pos, T const& val)
        {
            partitioned_vector_partition_[pos] = val;
        }

        /// Copy the value of \a val for the elements at positions \a pos in
        /// the partitioned_vector_partition container.
        ///
        /// \param pos   Positions of the elements in the
        ///              partitioned_vector_partition
        ///
        /// \param val   The value to be copied
        ///
        void set_values(std::vector<size_type> const& pos,
            std::vector<T> const& val)
        {
            HPX_ASSERT(pos.size() == val.size());
            HPX_ASSERT(pos.size() <= partitioned_vector_partition_.size());

            for (std::size_t i = 0; i != pos.size(); ++i)
                partitioned_vector_partition_[pos[i]] = val[i];
        }

        /// Remove all elements from the vector leaving the
        /// partitioned_vector_partition with size 0.
        ///
        void clear()
        {
            partitioned_vector_partition_.clear();
        }

        /// Macros to define HPX component actions for all exported functions.
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector, size);

//         HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector_partition, max_size);

        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector, resize);

//         HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector_partition, capacity);
//         HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector_partition, empty);
//         HPX_DEFINE_COMPONENT_ACTION(partitioned_vector_partition, reserve);

        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector, get_value);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector, get_values);

//         HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector_partition, front);
//         HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector_partition, back);
//         HPX_DEFINE_COMPONENT_ACTION(partitioned_vector_partition, assign);
//         HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector_partition, push_back);
//         HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector_partition, pop_back);

        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector, set_value);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector, set_values);

//         HPX_DEFINE_COMPONENT_ACTION(partitioned_vector_partition, clear);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector, get_copied_data);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partitioned_vector, set_data);
    };
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_PARTITIONED_VECTOR_DECLARATION(...)                      \
    HPX_REGISTER_VECTOR_DECLARATION_(__VA_ARGS__)                             \
/**/
#define HPX_REGISTER_VECTOR_DECLARATION_(...)                                 \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_VECTOR_DECLARATION_, HPX_UTIL_PP_NARG(__VA_ARGS__)       \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_VECTOR_DECLARATION_IMPL(type, name)                      \
    HPX_REGISTER_ACTION_DECLARATION(type::get_value_action,                   \
        BOOST_PP_CAT(__vector_get_value_action_, name));                      \
    HPX_REGISTER_ACTION_DECLARATION(type::get_values_action,                  \
        BOOST_PP_CAT(__vector_get_values_action_, name));                     \
    HPX_REGISTER_ACTION_DECLARATION(type::set_value_action,                   \
        BOOST_PP_CAT(__vector_set_value_action_, name));                      \
    HPX_REGISTER_ACTION_DECLARATION(type::set_values_action,                  \
        BOOST_PP_CAT(__vector_set_values_action_, name));                     \
    HPX_REGISTER_ACTION_DECLARATION(type::size_action,                        \
        BOOST_PP_CAT(__vector_size_action_, name));                           \
    HPX_REGISTER_ACTION_DECLARATION(type::resize_action,                      \
        BOOST_PP_CAT(__vector_resize_action_, name));                         \
    HPX_REGISTER_ACTION_DECLARATION(type::get_copied_data_action,             \
        BOOST_PP_CAT(__vector_get_copied_data_action_, name));                \
    HPX_REGISTER_ACTION_DECLARATION(type::set_data_action,                    \
        BOOST_PP_CAT(__vector_set_data_action_, name));                       \
/**/

#define HPX_REGISTER_VECTOR_DECLARATION_1(type)                               \
    HPX_REGISTER_VECTOR_DECLARATION_2(type, std::vector<type>)                \
/**/
#define HPX_REGISTER_VECTOR_DECLARATION_2(type, data)                         \
    HPX_REGISTER_VECTOR_DECLARATION_3(type, data, type)                       \
/**/
#define HPX_REGISTER_VECTOR_DECLARATION_3(type, data, name)                   \
    typedef ::hpx::server::partitioned_vector<type, data>                     \
        BOOST_PP_CAT(__partitioned_vector_, BOOST_PP_CAT(type, name));        \
    HPX_REGISTER_VECTOR_DECLARATION_IMPL(                                     \
        BOOST_PP_CAT(__partitioned_vector_, BOOST_PP_CAT(type, name)), name)  \
/**/

#define HPX_REGISTER_PARTITIONED_VECTOR(...)                                  \
    HPX_REGISTER_VECTOR_(__VA_ARGS__)                                         \
/**/
#define HPX_REGISTER_VECTOR_(...)                                             \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_VECTOR_, HPX_UTIL_PP_NARG(__VA_ARGS__)                   \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_VECTOR_IMPL(type, name)                                  \
    HPX_REGISTER_ACTION(type::get_value_action,                               \
        BOOST_PP_CAT(__vector_get_value_action_, name));                      \
    HPX_REGISTER_ACTION(type::get_values_action,                              \
        BOOST_PP_CAT(__vector_get_values_action_, name));                     \
    HPX_REGISTER_ACTION(type::set_value_action,                               \
        BOOST_PP_CAT(__vector_set_value_action_, name));                      \
    HPX_REGISTER_ACTION(type::set_values_action,                              \
        BOOST_PP_CAT(__vector_set_values_action_, name));                     \
    HPX_REGISTER_ACTION(type::size_action,                                    \
        BOOST_PP_CAT(__vector_size_action_, name));                           \
    HPX_REGISTER_ACTION(type::resize_action,                                  \
        BOOST_PP_CAT(__vector_resize_action_, name));                         \
    HPX_REGISTER_ACTION(type::get_copied_data_action,                         \
        BOOST_PP_CAT(__vector_get_copied_data_action_, name));                \
    HPX_REGISTER_ACTION(type::set_data_action,                                \
        BOOST_PP_CAT(__vector_set_data_action_, name));                       \
    typedef ::hpx::components::component<type> BOOST_PP_CAT(__vector_, name); \
    HPX_REGISTER_COMPONENT(BOOST_PP_CAT(__vector_, name))                     \
/**/

#define HPX_REGISTER_VECTOR_1(type)                                           \
    HPX_REGISTER_VECTOR_2(type, std::vector<type>)                            \
/**/
#define HPX_REGISTER_VECTOR_2(type, data)                                     \
    HPX_REGISTER_VECTOR_3(type, data, type)                                   \
/**/
#define HPX_REGISTER_VECTOR_3(type, data, name)                               \
    typedef ::hpx::server::partitioned_vector<type, data>                     \
        BOOST_PP_CAT(__partitioned_vector_, BOOST_PP_CAT(type, name));        \
    HPX_REGISTER_VECTOR_IMPL(                                                 \
        BOOST_PP_CAT(__partitioned_vector_, BOOST_PP_CAT(type, name)), name)  \
/**/

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    template <typename T, typename Data = std::vector<T> >
    class partitioned_vector_partition
      : public components::client_base<
            partitioned_vector_partition<T, Data>,
            server::partitioned_vector<T, Data>
        >
    {
    private:
        typedef hpx::server::partitioned_vector<T, Data> server_type;
        typedef hpx::components::client_base<
                partitioned_vector_partition<T, Data>,
                server::partitioned_vector<T, Data>
            > base_type;

    public:
        partitioned_vector_partition() {}

        partitioned_vector_partition(id_type const& gid)
          : base_type(gid)
        {}

        partitioned_vector_partition(hpx::shared_future<id_type> const& gid)
          : base_type(gid)
        {}

        // Return the pinned pointer to the underlying component
        std::shared_ptr<server::partitioned_vector<T, Data> > get_ptr() const
        {
            error_code ec(lightweight);
            return hpx::get_ptr<server::partitioned_vector<T, Data> >(
                this->get_id()).get(ec);
        }

        ///////////////////////////////////////////////////////////////////////
        //  Capacity related API's in partitioned_vector_partition client class

        /// Asynchronously return the size of the partitioned_vector_partition
        /// component.
        ///
        /// \return This returns size as the hpx::future of type size_type
        ///
        future<std::size_t> size_async() const
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::size_action>(this->get_id());
        }

        /// Return the size of the partitioned_vector_partition component.
        ///
        /// \return This returns size as the hpx::future of type size_type
        ///
        std::size_t size() const
        {
            return size_async().get();
        }

//         future<std::size_t> max_size_async() const
//         {
//             HPX_ASSERT(this->get_id());
//             return this->base_type::max_size_async(this->get_id());
//         }
//         std::size_t max_size() const
//         {
//             return max_size_async().get();
//         }

        /// Resize the partitioned_vector_partition component.
        /// If the \a val is not specified it use default constructor instead.
        ///
        /// \param n    New size of the partitioned_vector_partition
        /// \param val  Value to be copied if \a n is greater than the current
        ///             size
        ///
        void resize(std::size_t n, T const& val = T())
        {
            return resize_async(n, val).get();
        }

        /// Resize the partitioned_vector_partition component.
        /// If the \a val is not specified it use default constructor instead.
        ///
        /// \param n    New size of the partitioned_vector_partition
        /// \param val  Value to be copied if \a n is greater than the current
        ///             size
        ///
        /// \return This returns the hpx::future of type void which gets ready
        ///         once the operation is finished.
        ///
        future<void> resize_async(std::size_t n, T const& val = T())
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::resize_action>(
                this->get_id(), n, val);
        }

//         future<std::size_t> capacity_async() const
//         {
//             HPX_ASSERT(this->get_id());
//             return this->base_type::capacity_async(this->get_id());
//         }
//         std::size_t capacity() const
//         {
//             return capacity_async().get();
//         }

//         future<bool> empty_async() const
//         {
//             HPX_ASSERT(this->get_id());
//             return this->base_type::empty_async(this->get_id());
//         }
//         bool empty() const
//         {
//             return empty_async().get();
//         }

//         void reserve(std::size_t n)
//         {
//             HPX_ASSERT(this->get_id());
//             this->base_type::reserve_async(this->get_id(), n).get();
//         }

        //  Element Access API's in Client class

        /// Returns the value at position \a pos in the partitioned_vector_partition
        /// component.
        ///
        /// \param pos  Position of the element in the partitioned_vector_partition
        ///
        /// \return Returns the value of the element at position represented
        ///         by \a pos
        ///
        T get_value(launch::sync_policy, std::size_t pos) const
        {
            return get_value(pos).get();
        }
#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        T get_value_sync(std::size_t pos) const
        {
            return get_value(launch::sync, pos);
        }
#endif

        /// Return the element at the position \a pos in the
        /// partitioned_vector_partition container.
        ///
        /// \param pos Position of the element in the partitioned_vector_partition
        ///
        /// \return This returns the value as the hpx::future
        ///
        future<T> get_value(std::size_t pos) const
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::get_value_action>(
                this->get_id(), pos);
        }

        /// Returns the value at position \a pos in the partitioned_vector_partition
        /// component.
        ///
        /// \param pos  Position of the element in the partitioned_vector_partition
        ///
        /// \return Returns the value of the element at position represented
        ///         by \a pos
        ///
        std::vector<T> get_values(launch::sync_policy,
            std::vector<std::size_t> const& pos) const
        {
            return get_values(pos).get();
        }
#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        std::vector<T> get_values_sync(std::vector<std::size_t> const& pos) const
        {
            return get_values(launch::sync, pos);
        }
#endif

        /// Return the element at the position \a pos in the
        /// partitioned_vector_partition container.
        ///
        /// \param pos Position of the element in the partitioned_vector_partition
        ///
        /// \return This returns the value as the hpx::future
        ///
        future<std::vector<T> >
        get_values(std::vector<std::size_t> const& pos) const
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::get_values_action>(
                this->get_id(), pos);
        }

//         future<T> front_async() const
//         {
//             HPX_ASSERT(this->get_id());
//             return this->base_type::front_async(this->get_id());
//         }
//         T front() const
//         {
//             HPX_ASSERT(this->get_id());
//             return front_async().get();
//         }

//         future<T> back_async() const
//         {
//             HPX_ASSERT(this->get_id());
//             return this->base_type::back_async(this->get_id());
//         }
//         T back() const
//         {
//             return back_async().get();
//         }

        //  Modifiers API's in client class
//         void assign(std::size_t n, T const& val)
//         {
//             HPX_ASSERT(this->get_id());
//             this->base_type::assign_async(this->get_id(), n, val).get();
//         }

//         template <typename T_>
//         void push_back(T_ && val)
//         {
//             HPX_ASSERT(this->get_id());
//             this->base_type::push_back_async(
//                 this->get_id(), std::forward<T_>(val)).get();
//         }

//         void pop_back()
//         {
//             HPX_ASSERT(this->get_id());
//             this->base_type::pop_back_async(this->get_id()).get();
//         }

        /// Copy the value of \a val in the element at position
        /// \a pos in the partitioned_vector_partition container.
        ///
        /// \param pos   Position of the element in the partitioned_vector_partition
        /// \param val   The value to be copied
        ///
        template <typename T_>
        void set_value(launch::sync_policy, std::size_t pos, T_ && val)
        {
            set_value(pos, std::forward<T_>(val)).get();
        }
#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        template <typename T_>
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void set_value_sync(std::size_t pos, T_ && val)
        {
            set_value(launch::sync, pos, std::forward<T_>(val));
        }
#endif

        /// Copy the value of \a val in the element at position
        /// \a pos in the partitioned_vector_partition component.
        ///
        /// \param pos  Position of the element in the partitioned_vector_partition
        /// \param val  Value to be copied
        ///
        /// \return This returns the hpx::future of type void
        ///
        template <typename T_>
        future<void> set_value(std::size_t pos, T_ && val)
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::set_value_action>(
                this->get_id(), pos, std::forward<T_>(val));
        }

        /// Copy the value of \a val in the element at position
        /// \a pos in the partitioned_vector_partition container.
        ///
        /// \param pos   Position of the element in the partitioned_vector_partition
        /// \param val   The value to be copied
        ///
        void set_values(launch::sync_policy,
            std::vector<std::size_t> const& pos, std::vector<T> const& val)
        {
            set_values(pos, val).get();
        }
#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void set_values_sync(std::vector<std::size_t> const& pos,
            std::vector<T> const& val)
        {
            set_values(launch::sync, pos, val);
        }
#endif

        /// Copy the value of \a val in the element at position
        /// \a pos in the partitioned_vector_partition component.
        ///
        /// \param pos  Position of the element in the partitioned_vector_partition
        /// \param val  Value to be copied
        ///
        /// \return This returns the hpx::future of type void
        ///
        future<void> set_values(std::vector<std::size_t> const& pos,
            std::vector<T> const& val)
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::set_values_action>(
                this->get_id(), pos, val);
        }

//         void clear()
//         {
//             HPX_ASSERT(this->get_id());
//             this->base_type::clear_async(this->get_id()).get();
//         }

        /// Returns a copy of the data owned by the partitioned_vector_partition
        /// component.
        ///
        /// \return This returns the data of the partitioned_vector_partition
        ///
        typename server_type::data_type get_copied_data(launch::sync_policy) const
        {
            return get_copied_data().get();
        }
#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        typename server_type::data_type get_copied_data_sync() const
        {
            return get_copied_data(launch::sync);
        }
#endif

        /// Returns a copy of the data owned by the partitioned_vector_partition
        /// component.
        ///
        /// \return This returns the data as an hpx::future
        ///
        hpx::future<typename server_type::data_type> get_copied_data() const
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::get_copied_data_action>(
                this->get_id());
        }

        /// Updates the data owned by the partition_vector
        /// component.
        ///
        /// \return This returns the data of the partition_vector
        ///
        void set_data(launch::sync_policy,
            typename server_type::data_type && other) const
        {
            set_data(std::move(other)).get();
        }
#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void set_data_sync(typename server_type::data_type && other) const
        {
            set_data(launch::sync, std::move(other));
        }
#endif

        /// Updates the data owned by the partition_vector
        /// component.
        ///
        /// \return This returns the hpx::future of type void
        ///
        hpx::future<void> set_data(typename server_type::data_type && other) const
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::set_data_action>(
                this->get_id(), std::move(other) );
        }
   };
}

#endif
