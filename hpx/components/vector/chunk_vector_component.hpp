//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file chunk_vector_component.hpp

#ifndef HPX_CHUNK_VECTOR_COMPONENT_HPP
#define HPX_CHUNK_VECTOR_COMPONENT_HPP

/// \file hpx/components/vector/chunk_vector_component.hpp
///
/// \brief The chunk_vector as the hpx component is defined here.
///
/// The chunk_vector is the wrapper to the stl vector class except all API's
/// are defined as component action. All the API's in stubs classes are
/// asynchronous API which return the futures.

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>

#include <iostream>
#include <tuple>
#include <vector>
#include <string>

/** @brief Defines the type of value stored by elements in the chunk_vector.*/
#define VALUE_TYPE double

namespace hpx { namespace server
{
    /// \brief This is the basic wrapper class for stl vector.
    ///
    /// This contain the implementation of the chunk_vector's component
    /// functionality.

    class HPX_COMPONENT_EXPORT chunk_vector
      : public hpx::components::locking_hook<
            hpx::components::simple_component_base<chunk_vector>
        >
    {
    private:
        /** @brief It is the std::vector of VALUE_TYPE. */
        std::vector<VALUE_TYPE> chunk_vector_;

    public:
        typedef std::size_t size_type;

        ///////////////////////////////////////////////////////////////////////
        // Constructors
        ///////////////////////////////////////////////////////////////////////

        /** @brief Default Constructor which create chunk_vector with size 0.
            */
        chunk_vector()
        {}

//            explicit chunk_vector(size_type chunk_size)
//                : chunk_vector_(chunk_size, VALUE_TYPE()) {}

        /** @brief Constructor which create and initialize chunk_vector with
         *          all elements as \a val.
         *
         *  @param chunk_size The size of vector
         *  @param val Default value for the elements in chunk_vector
         */
        chunk_vector(size_type chunk_size, VALUE_TYPE const& val)
          : chunk_vector_(chunk_size, val)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Capacity Related API's in the server class
        ///////////////////////////////////////////////////////////////////////

        /** @brief Return the size of this chunk
         *
         *  @return Return the number of elements in the chunk_vector
         */
        size_type size() const
        {
            return chunk_vector_.size();
        }

        /** @brief Compute the maximum size of chunk_vector in terms of
         *          number of elements.
         *
         *  @return Return maximum number of elements the chunk_vector can
         *           hold
         */
        size_type max_size() const
        {
            return chunk_vector_.max_size();
        }

        /** @brief Resize the chunk_vector so that it contain \a n elements.
         *
         *  @param n    new size of the chunk_vector
         *  @param val  value to be copied if \a n is greater than the
         *               current size
         */
        void resize(size_type n, VALUE_TYPE const& val)
        {
            //chunk_vector_.resize(n, val);
        }

        /** @brief Compute the size of currently allocated storage capacity
         *          for chunk_vector.
         *
         *  @return Returns capacity of chunk_vector, expressed in terms of
         *           elements
         */
        size_type capacity() const
        {
            return chunk_vector_.capacity();
        }

        /** @brief Return whether the chunk_vector is empty.
         *
         *  @return Return true if chunk_vector size is 0, false otherwise
         */
        bool empty() const
        {
            return chunk_vector_.empty();
        }

        /** @brief Request the change in chunk_vector capacity so that it
         *          can hold \a n elements. Throws the \a hpx::length_error
         *          exception.
         *
         *  This function request chunk_vector capacity should be at least
         *   enough to contain n elements. If n is greater than current
         *   chunk_vector capacity, the function causes the chunk_vector to
         *   reallocate its storage increasing its capacity to n (or greater).
         *  In other cases the chunk_vector capacity does not got affected.
         *  It does not change the chunk_vector size.
         *
         * @param n minimum capacity of chunk_vector
         *
         */
        void reserve(size_type n)
        {
            chunk_vector_.reserve(n);
        }

        ///////////////////////////////////////////////////////////////////////
        // Element access API's
        ///////////////////////////////////////////////////////////////////////

        /** @brief Return the element at the position \a pos in the
         *          chunk_vector container. It does not throw the
         *          exception.
         *
         *  @param pos Position of the element in the chunk_vector [Note the
         *              first position in the chunk_vector is 0]
         *  @return Return the value of the element at position represented
         *           by \a pos [Note that this is not the reference to the
         *           element]
         */
        VALUE_TYPE get_value(size_type pos) const
        {
            return chunk_vector_[pos];
        }

        /** @brief Access the value of first element in the chunk_vector.
         *
         *  Calling the function on empty container cause undefined behavior.
         *
         * @return Return the value of the first element in the chunk_vector
         */
        VALUE_TYPE front() const
        {
            return chunk_vector_.front();
        }

        /** @brief Access the value of last element in the chunk_vector.
         *
         *  Calling the function on empty container cause undefined behavior.
         *
         * @return Return the value of the last element in the chunk_vector
         */
        VALUE_TYPE back() const
        {
            return chunk_vector_.back();
        }

        ///////////////////////////////////////////////////////////////////////
        // Modifiers API's in server class
        ///////////////////////////////////////////////////////////////////////

        /** @brief Assigns new contents to the chunk_vector, replacing its
         *          current contents and modifying its size accordingly.
         *
         * @param n     new size of chunk_vector
         * @param val   Value to fill the container with
         */
        void assign(size_type n, VALUE_TYPE const& val)
        {
            chunk_vector_.assign(n, val);
        }

        /** @brief Add new element at the end of chunk_vector. The added
         *          element contain the \a val as value.
         *
         * @param val Value to be copied to new element
         */
        void push_back(VALUE_TYPE const& val)
        {
            chunk_vector_.push_back(val);
        }

        /** @brief Remove the last element from chunk_vector effectively
         *          reducing the size by one. The removed element is destroyed.
         */
        void pop_back()
        {
            chunk_vector_.pop_back();
        }

        //  This API is required as we do not returning the reference to the
        //  element in Any API.

        /** @brief Copy the value of \a val in the element at position
         *          \a pos in the chunk_vector container. It throws the
         *          \a hpx::out_of_range exception.
         *
         *  @param pos   Position of the element in the chunk_vector [Note
         *                the first position in the chunk_vector is 0]
         *  @param val   The value to be copied
         *
         */
        void set_value(size_type pos, VALUE_TYPE const& val)
        {
            chunk_vector_[pos] = val;
        }

        //TODO deprecate it
        /** @brief Remove all elements from the vector leaving the
            *          chunk_vector with size 0.
            */
        void clear()
        {
            chunk_vector_.clear();
        }

        ///////////////////////////////////////////////////////////////////////
        // Algorithm API's
        ///////////////////////////////////////////////////////////////////////

        /** @brief Apply the function \a f to each element in the range
         *          [first, last).
         *
         *  @param first    Initial position of the element in the sequence
         *                  [Note the first position in the chunk_vector
         *                  is 0]
         *  @param last     Final position of the element in the sequence
         *                  [Note the last element is not inclusive in the
         *                  range[first, last)]
         *  @param f        Unary function (either function pointer or move
         *                  constructible function object) that accept an
         *                  element in the range as argument.
         */
        void for_each(size_type first, size_type last,
            hpx::util::function<void(double&)> f)
        {
            HPX_ASSERT(first < chunk_vector_.size());
            HPX_ASSERT(last < chunk_vector_.size());

            parallel::for_each(parallel::par, chunk_vector_.begin() + first,
                chunk_vector_.begin() + last, f);
        }

        /** @brief Macro to define \a size function as HPX component action
        *           type.
        */
        HPX_DEFINE_COMPONENT_CONST_ACTION(chunk_vector, size);
        /** @brief Macro to define \a max_size function as HPX component
            *          action type.
            */
        HPX_DEFINE_COMPONENT_CONST_ACTION(chunk_vector, max_size);
        /** @brief Macro to define \a resize function as HPX component
            *          action type.
            */
        HPX_DEFINE_COMPONENT_ACTION(chunk_vector, resize);
        /** @brief Macro to define \a capacity function as HPX component
            *          action type.
            */
        HPX_DEFINE_COMPONENT_CONST_ACTION(chunk_vector, capacity);
        /** @brief Macro to define \a empty function as HPX component action
            *          type.
            */
        HPX_DEFINE_COMPONENT_CONST_ACTION(chunk_vector, empty);
        /** @brief Macro to define \a reserve function as HPX component
            *          action type.
            */
        HPX_DEFINE_COMPONENT_ACTION(chunk_vector, reserve);

        /** @brief Macro to define \a get_value function as HPX component
            *          action type.
            */
        HPX_DEFINE_COMPONENT_CONST_ACTION(chunk_vector, get_value);
        /** @brief Macro to define \a front function as HPX component
            *          action type.
            */
        HPX_DEFINE_COMPONENT_CONST_ACTION(chunk_vector, front);
        /** @brief Macro to define \a back function as HPX component
            *          action type.
            */
        HPX_DEFINE_COMPONENT_CONST_ACTION(chunk_vector, back);

        /** @brief Macro to define \a assign function as HPX component
            *          action type.
            */
        HPX_DEFINE_COMPONENT_ACTION(chunk_vector, assign);
        /** @brief Macro to define \a push_back function as HPX component
            *          action type.
            */
        HPX_DEFINE_COMPONENT_ACTION(chunk_vector, push_back);
        /** @brief Macro to define \a pop_back function as HPX component
            *          action type.
            */
        HPX_DEFINE_COMPONENT_ACTION(chunk_vector, pop_back);
        /** @brief Macro to define \a set_value function as HPX component
            *          action type.
            */
        HPX_DEFINE_COMPONENT_ACTION(chunk_vector, set_value);
        /** @brief Macro to define \a clear function as HPX component action
            *          type.
            */
        HPX_DEFINE_COMPONENT_ACTION(chunk_vector, clear);

        /** @brief Macro to define \a chunk_for_each function as HPX
            *          component action type.
            */
        HPX_DEFINE_COMPONENT_ACTION(chunk_vector, for_each);
    };
}}

//Capacity related action declaration
/** @brief Macro to register \a size component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::size_action,
    chunk_vector_size_action);
/** @brief Macro to register \a max_size component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::max_size_action,
    chunk_vector_max_size_action);
/** @brief Macro to register \a resize component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::resize_action,
    chunk_vector_resize_action);
/** @brief Macro to register \a capacity component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::capacity_action,
    chunk_vector_capacity_action);
/** @brief Macro to register \a empty component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::empty_action,
    chunk_vector_empty_action);
/** @brief Macro to register \a reserve component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::reserve_action,
    chunk_vector_reserve_action);

// Element access component action declaration
/** @brief Macro to register \a get_value component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::get_value_action,
    chunk_vector_get_value_action);
/** @brief Macro to register \a front component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::front_action,
    chunk_vector_front_action);
/** @brief Macro to register \a back component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::back_action,
    chunk_vector_back_action);

//Modifiers component action declaration
/** @brief Macro to register \a assign component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::assign_action,
    chunk_vector_assign_action);
/** @brief Macro to register \a push_back component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::push_back_action,
    chunk_vector_push_back_action);
/** @brief Macro to register \a pop_back component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::pop_back_action,
    chunk_vector_pop_back_action);
/** @brief Macro to register \a set_value component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::set_value_action,
    chunk_vector_set_value_action);
/** @brief Macro to register \a clear component action type with HPX AGAS.*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::clear_action,
    chunk_vector_clear_action);

//Algorithm API's component action declaration
/** @brief Macro to register \a for_each component action type with HPX
*           AGAS.
*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::chunk_vector::for_each_action,
    chunk_vector_for_each_action);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace stubs
{
    /** @brief This is the low level interface to the chunk_vector.
     *
     * This class contain the implementation of the low level interface to
     *  the chunk_vector. All the function provides asynchronous interface
     *  which mean every function does not block the caller and returns the
     *  future as return value.
     */
    template <typename T>
    struct chunk_vector : hpx::components::stub_base<server::chunk_vector>
    {
    private:
        typedef hpx::server::chunk_vector base_type;

    public:
        typedef base_type::size_type            size_type;

        /** @brief Calculate the size of the chunk_vector component.
         *
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *
         *  @return This return size as the hpx::future of type size_type
         */
        static future<size_type> size_async(id_type const& gid)
        {
            return hpx::async<base_type::size_action>(gid);
        }

        /** @brief Calculate the maximum size of the chunk_vector component
         *          in terms of number of elements.
         *
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *
         *  @return This return maximum size as the hpx::future of size_type
         */
        static future<size_type> max_size_async(id_type const& gid)
        {
            return hpx::async<base_type::max_size_action>(gid);
        }

        /** @brief Resize the chunk_vector component. If the \a val is not
         *          it use default constructor instead.
         *
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *  @param n    New size of the chunk_vector
         *  @param val  Value to be copied if \a n is greater than the
         *               current size
         *
         *  @return This return the hpx::future of type void [The void return
         *           type can help to check whether the action is completed or
         *           not]
         */
        static future<void> resize_async(id_type const& gid,
            size_type n, T const& val = T())
        {
            return hpx::async<base_type::resize_action>(gid, n, val);
        }

        /** @brief Calculate the capacity of the chunk_vector component.
         *
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *
         *  @return This return capacity as the hpx::future of size_type
         */
        static future<size_type> capacity_async(id_type const& gid)
        {
            return hpx::async<base_type::capacity_action>(gid);
        }

        /** @brief Check whether chunk_vector component is empty.
         *
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *
         *  @return This function return result as the hpx::future of type
         *           bool
         */
        static future<bool> empty_async(id_type const& gid)
        {
            return hpx::async<base_type::empty_action>(gid);
        }

        /** @brief Reserve the storage space for chunk_vector component.
         *          Throws the \a hpx::length_error exception.
         *
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *  @param n    Minimum size of the chunk_vector
         *
         *  @exception hpx::length_error If \a n is greater than maximum
         *              size then function throw \a hpx::length_error
         *              exception.
         *
         *  @return This return the hpx::future of type void [The void return
         *           type can help to check whether the action is completed
         *           or not]
         */
        static future<void> reserve_async(id_type const& gid, size_type n)
        {
            return hpx::async<base_type::reserve_action>(gid, n);
        }

        ///////////////////////////////////////////////////////////////////////
        //  Element Access API's in stubs class
        ///////////////////////////////////////////////////////////////////////

        /** @brief Return the value at position \a pos in the chunk_vector
         *          component. It throws the \a hpx::out_of_range exception.
         *
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *  @param pos  Position of the element in the chunk_vector [Note
         *               the first position in the chunk_vector is 0]
         *
         * @exception hpx::out_of_range The \a pos is bound checked and if
         *             \a pos is out of bound then it throws the
         *             \a hpx::out_of_range exception.
         *
         *  @return This return value as the hpx::future
         */
        static future<T> get_value_async(id_type const& gid, size_type pos)
        {
            return hpx::async<base_type::get_value_action>(gid, pos);
        }

        /** @brief Access the value of first element in chunk_vector
         *          component.
         *
         *  @param gid  The global id of the chunk_vector component
         *               register with HPX
         *
         *  @return This return value as the hpx::future
         */
        static future<T> front_async(id_type const& gid)
        {
            return hpx::async<base_type::front_action>(gid);
        }

        /** @brief Access the value of last element in chunk_vector
         *          component.
         *
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *
         *  @return This return value as the hpx::future
         */
        static future<T> back_async(id_type const& gid)
        {
            return hpx::async<base_type::back_action>(gid);
        }

        ///////////////////////////////////////////////////////////////////////
        //  Modifiers API's in stubs class
        ///////////////////////////////////////////////////////////////////////

        /** @brief Assign the new content to the elements in chunk_vector
         *          component, replacing the current content and modifying
         *          the size accordingly.
         *
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *  @param n    New size of the chunk_vector
         *  @param val  Value to fill the container with
         *
         *  @return This return the hpx::future of type void [The void
         *           return type can help to check whether the action is
         *           completed or not]
         */
        static future<void>
        assign_async(id_type const& gid, size_type n, T const& val)
        {
            return hpx::async<base_type::assign_action>(gid, n, val);
        }

        /** @brief Add the new element at the end of chunk_vector component.
         *
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *  @param val  Value to be copied to new element
         *
         *  @return This return the hpx::future of type void [The void
         *           return type can help to check whether the action is
         *           completed or not]
         */
        template <typename T_>
        static future<void> push_back_async(id_type const& gid, T_ && val)
        {
            return hpx::async<base_type::push_back_action>(
                gid, std::forward<T_>(val));
        }

        /** @brief Remove the last element from chunk_vector effectively
         *          reducing the size by one. The removed element is destroyed.
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *  @return This return the hpx::future of type void [The void
         *           return type can help to check whether the action is
         *           completed or not].
         */
        static future<void> pop_back_async(id_type const& gid)
        {
            return hpx::async<base_type::pop_back_action>(gid);
        }

        /** @brief Copy the value of \a val in the element at position
         *          \a pos in the chunk_vector component. It throws the
         *          \a hpx::out_of_range exception.
         *
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *  @param pos  Position of the element in the chunk_vector [Note
         *               the first position in the chunk_vector is 0].
         *  @param val  Value to be copied
         *
         *  @exception hpx::out_of_range The \a pos is bound checked and if
         *             \a pos is out of bound then it throws the
         *             \a hpx::out_of_range exception.
         *
         *  @return This return the hpx::future of type void [The void
         *           return type can help to check whether the action is
         *           completed or not].
         */
        template <typename T_>
        static future<void>
        set_value_async(id_type const& gid, size_type pos, T_ && val)
        {
            return hpx::async<base_type::set_value_action>(
                gid, pos, std::forward<T_>(val));
        }

        /** @brief Remove all elements from the vector leaving the
         *          chunk_vector with size 0.
         *  @param gid  The global id of the chunk_vector component register
         *               with HPX
         *  @return This return the hpx::future of type void [The void
         *           return type can help to check whether the action is
         *           completed or not].
         */
        static future<void> clear_async(id_type const& gid)
        {
            return hpx::async<base_type::clear_action>(gid);
        }

        ///////////////////////////////////////////////////////////////////////
        // Algorithm API's in Stubs class
        ///////////////////////////////////////////////////////////////////////

        /** @brief Apply the function \a fn to each element in the range
         *          [first, last) in chunk_vector component.
         *
         *  @param gid      The global id of the chunk_vector component
         *                   register with HPX
         *  @param first    Initial position of the element in the sequence
         *                   [Note the first position in the chunk_vector
         *                     is 0].
         *  @param last     Final position of the element in the sequence
         *                  [Note the last element is not inclusive in the
         *                   range [first, last)].
         *  @param fn       Unary function (either function pointer or move
         *                   constructible function object) that accept an
         *                   element in the range as argument.
         * @return This return the hpx::future of type void [The void return
         *          type can help to check whether the action is completed
         *          or not]
         */
        static future<void>
        for_each_async(id_type const& gid, size_type first,
            size_type last, hpx::util::function<void(double&)> fn)
        {
            typedef base_type::for_each_action action_type;
            return hpx::async<action_type>(gid, first, last, fn);
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    template <typename T>
    class chunk_vector
      : public components::client_base<chunk_vector<T>, stubs::chunk_vector<T> >
    {
    private:
        typedef hpx::components::client_base<
                chunk_vector<T>, stubs::chunk_vector<T>
            > base_type;

    public:
        chunk_vector() {}

        chunk_vector(id_type const& gid)
          : base_type(gid)
        {}

        chunk_vector(hpx::shared_future<id_type> const& gid)
          : base_type(gid)
        {}

        //  Capacity related API's in chunk_vector client class
        future<std::size_t> size_async() const
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::size_async(this->get_gid());
        }
        std::size_t size() const
        {
            return size_async().get();
        }

        future<std::size_t> max_size_async() const
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::max_size_async(this->get_gid());
        }
        std::size_t max_size() const
        {
            return max_size_async().get();
        }

        void resize(std::size_t n, T const& val = 0)
        {
            HPX_ASSERT(this->get_gid());
            this->base_type::resize_async(this->get_gid(), n, val).get();
        }

        future<std::size_t> capacity_async() const
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::capacity_async(this->get_gid());
        }
        std::size_t capacity() const
        {
            return capacity_async().get();
        }

        future<bool> empty_async() const
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::empty_async(this->get_gid());
        }
        bool empty() const
        {
            return empty_async().get();
        }

        void reserve(std::size_t n)
        {
            HPX_ASSERT(this->get_gid());
            this->base_type::reserve_async(this->get_gid(), n).get();
        }

        //  Element Access API's in Client class
        future<T> get_value_async(std::size_t pos) const
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::get_value_async(this->get_gid(), pos);
        }
        T get_value(std::size_t pos) const
        {
            return get_value_async(pos).get();
        }

        future<T> front_async() const
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::front_async(this->get_gid());
        }
        T front() const
        {
            HPX_ASSERT(this->get_gid());
            return front_async().get();
        }

        future<T> back_async() const
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::back_async(this->get_gid());
        }
        T back() const
        {
            return back_async().get();
        }

        //  Modifiers API's in client class
        void assign(std::size_t n, T const& val)
        {
            HPX_ASSERT(this->get_gid());
            this->base_type::assign_async(this->get_gid(), n, val).get();
        }

        template <typename T_>
        void push_back(T_ && val)
        {
            HPX_ASSERT(this->get_gid());
            this->base_type::push_back_async(
                this->get_gid(), std::forward<T_>(val)).get();
        }

        void pop_back()
        {
            HPX_ASSERT(this->get_gid());
            this->base_type::pop_back_async(this->get_gid()).get();
        }

        template <typename T_>
        void set_value(std::size_t pos, T_ && val)
        {
            HPX_ASSERT(this->get_gid());
            this->base_type::set_value_async(
                this->get_gid(), pos, std::forward<T_>(val)).get();
        }

        void clear()
        {
            HPX_ASSERT(this->get_gid());
            this->base_type::clear_async(this->get_gid()).get();
        }
    };
}

#endif
