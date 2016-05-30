//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file partition_unordered_map_component.hpp

#if !defined(HPX_PARTITION_UNORDERED_MAP_COMPONENT_NOV_11_2014_0853PM)
#define HPX_PARTITION_UNORDERED_MAP_COMPONENT_NOV_11_2014_0853PM

/// \file hpx/components/unordered/partition_unordered_map_component.hpp
///
/// \brief The partition_unordered_map as the hpx component is defined here.
///
/// The partition_unordered_map is the wrapper to the stl unordered_map class
/// except all API'are defined as component action. All the API's in client
/// classes are asynchronous API which return the futures.

#include <hpx/config.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/lcos/reduce.hpp>
#include <hpx/runtime/actions/basic_action.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/get_ptr.hpp>
#include <hpx/util/assert.hpp>

#include <boost/preprocessor/cat.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace hpx { namespace server
{
    /// \brief This is the basic wrapper class for stl unordered_map.
    ///
    /// This contain the implementation of the partition_unordered_map's
    /// component functionality.
    template <typename Key, typename T, typename Hash = std::hash<Key>,
        typename KeyEqual = std::equal_to<Key> >
    class partition_unordered_map
      : public components::locking_hook<
            hpx::components::simple_component_base<
                partition_unordered_map<Key, T, Hash, KeyEqual> > >
    {
    public:
        typedef std::unordered_map<Key, T, Hash, KeyEqual> data_type;

        typedef typename data_type::size_type size_type;
        typedef typename data_type::iterator iterator_type;
        typedef typename data_type::const_iterator const_iterator_type;

        typedef components::locking_hook<
                hpx::components::simple_component_base<
                    partition_unordered_map<Key, T, Hash, KeyEqual> > >
            base_type;

    private:
        data_type partition_unordered_map_;

    public:
        ///////////////////////////////////////////////////////////////////////
        // Constructors
        ///////////////////////////////////////////////////////////////////////

        /// Default Constructor which create partition_unordered_map
        /// with size 0.
        partition_unordered_map()
        {
        }

        explicit partition_unordered_map(size_type bucket_count)
          : partition_unordered_map_(bucket_count)
        {}

        partition_unordered_map(size_type bucket_count, Hash const& hash,
                KeyEqual const& equal)
          : partition_unordered_map_(bucket_count, hash, equal)
        {}

        // support components::copy
        partition_unordered_map(partition_unordered_map const& rhs)
          : base_type(rhs),
            partition_unordered_map_(rhs.partition_unordered_map_)
        {}

        partition_unordered_map& operator=(partition_unordered_map const& rhs)
        {
            if (this != &rhs)
            {
                this->base_type::operator=(rhs);
                partition_unordered_map_ = rhs.partition_unordered_map_;
            }
            return *this;
        }

        partition_unordered_map(partition_unordered_map && rhs)
          : base_type(std::move(rhs)),
            partition_unordered_map_(std::move(rhs.partition_unordered_map_))
        {}

        partition_unordered_map& operator=(partition_unordered_map && rhs)
        {
            if (this != &rhs)
            {
                this->base_type::operator=(std::move(rhs));
                partition_unordered_map_ = std::move(rhs.partition_unordered_map_);
            }
            return *this;
        }

        /// Duplicate the copy method for action naming
        data_type get_copied_data() const
        {
            return partition_unordered_map_;
        }
        void set_copied_data(data_type && d)
        {
            partition_unordered_map_ = std::move(d);
        }

        ///////////////////////////////////////////////////////////////////////
        iterator_type begin()
        {
            return partition_unordered_map_.begin();
        }
        const_iterator_type begin() const
        {
            return partition_unordered_map_.begin();
        }
        const_iterator_type cbegin() const
        {
            return partition_unordered_map_.cbegin();
        }

        iterator_type end()
        {
            return partition_unordered_map_.end();
        }
        const_iterator_type end() const
        {
            return partition_unordered_map_.end();
        }
        const_iterator_type cend() const
        {
            return partition_unordered_map_.cend();
        }

        ///////////////////////////////////////////////////////////////////////
        // Capacity Related API's in the server class
        ///////////////////////////////////////////////////////////////////////

        /// Returns the number of elements
        size_type size() const
        {
            return partition_unordered_map_.size();
        }

        /// Returns the maximum possible number of elements
        size_type max_size() const
        {
            return partition_unordered_map_.max_size();
        }

        /// Returns the number of elements that the container has currently
        /// allocated space for.
        size_type capacity() const
        {
            return partition_unordered_map_.capacity();
        }

        /// Checks if the container has no elements, i.e. whether
        /// begin() == end().
        bool empty() const
        {
            return partition_unordered_map_.empty();
        }

        ///////////////////////////////////////////////////////////////////////
        // Element access API's
        ///////////////////////////////////////////////////////////////////////

        /// Return the element at the position \a pos in the partition_unordered_map
        /// container.
        ///
        /// \param pos Position of the element in the partition_unordered_map
        ///
        /// \return Return the value of the element at position represented
        ///         by \a pos.
        ///
        struct erase_on_exit
        {
            erase_on_exit(data_type& m, typename data_type::iterator& it)
              : m_(m), it_(it)
            {}
            ~erase_on_exit()
            {
                m_.erase(it_);
            }

            data_type& m_;
            typename data_type::iterator& it_;
        };

        T get_value(Key const& key, bool erase)
        {
            typename data_type::iterator it = partition_unordered_map_.find(key);
            if (it == partition_unordered_map_.end())
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "partition_unordered_map::get_value",
                    "unable to find requested key in this partition of the "
                    "unordered_map");
            }

            if (!erase)
                return it->second;

            erase_on_exit t(partition_unordered_map_, it);
            return it->second;
        }

        /// Return the element at the position \a pos in the partition_unordered_map
        /// container.
        ///
        /// \param pos Positions of the elements in the partition_unordered_map
        ///
        /// \return Return the values of the elements at position represented
        ///         by \a pos.
        ///
        std::vector<T> get_values(std::vector<Key> const& keys)
        {
            std::vector<T> result;
            result.reserve(keys.size());

            for (std::size_t i = 0; i != keys.size(); ++i)
            {
                typename data_type::iterator it =
                    partition_unordered_map_.find(keys[i]);
                if (it == partition_unordered_map_.end())
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "partition_unordered_map::get_values",
                        "unable to find requested key in this partition of the "
                        "unordered_map");
                    break;
                }
                result.push_back(it->second);
            }
            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        // Modifiers API's in server class
        ///////////////////////////////////////////////////////////////////////

        /// Copy the value of \a val in the element at position \a pos in the
        /// partition_unordered_map container.
        ///
        /// \param pos   Position of the element in the partition_unordered_map
        ///
        /// \param val   The value to be copied
        ///
        void set_value(Key const& pos, T const& val)
        {
            partition_unordered_map_[pos] = val;
        }

        /// Copy the value of \a val for the elements at positions \a pos in
        /// the partition_unordered_map container.
        ///
        /// \param pos   Positions of the elements in the partition_unordered_map
        ///
        /// \param val   The value to be copied
        ///
        void set_values(std::vector<Key> const& keys,
            std::vector<T> const& val)
        {
            HPX_ASSERT(keys.size() == val.size());
            HPX_ASSERT(keys.size() <= partition_unordered_map_.size());

            for (std::size_t i = 0; i != keys.size(); ++i)
                partition_unordered_map_[keys[i]] = val[i];
        }

        /// Remove all elements from the vector leaving the
        /// partition_unordered_map with size 0.
        ///
        void clear()
        {
            partition_unordered_map_.clear();
        }

        /// Erase the given element
        std::size_t erase(Key const& key)
        {
            return partition_unordered_map_.erase(key);
        }

        /// Macros to define HPX component actions for all exported functions.
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partition_unordered_map, size);

        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partition_unordered_map, get_value);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partition_unordered_map, get_values);

        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partition_unordered_map, set_value);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partition_unordered_map, set_values);

        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partition_unordered_map, erase);

        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partition_unordered_map, get_copied_data);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(partition_unordered_map, set_copied_data);
    };
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_UNORDERED_MAP_DECLARATION(...)                           \
    HPX_REGISTER_UNORDERED_MAP_DECLARATION_(__VA_ARGS__)                      \
/**/
#define HPX_REGISTER_UNORDERED_MAP_DECLARATION_(...)                          \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_UNORDERED_MAP_DECLARATION_, HPX_UTIL_PP_NARG(__VA_ARGS__)\
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_UNORDERED_MAP_DECLARATION_2(key, type)                   \
    HPX_REGISTER_UNORDERED_MAP_DECLARATION_5(key, type, std::hash<key>,       \
        std::equal_to<key>, type)                                             \
/**/
#define HPX_REGISTER_UNORDERED_MAP_DECLARATION_3(key, type, hash)             \
    HPX_REGISTER_UNORDERED_MAP_DECLARATION_5(key, type, hash,                 \
        std::equal_to<key>, type)                                             \
/**/
#define HPX_REGISTER_UNORDERED_MAP_DECLARATION_4(key, type, hash, equal)      \
    HPX_REGISTER_UNORDERED_MAP_DECLARATION_5(key, type, hash, equal, type)    \
/**/

#define HPX_REGISTER_UNORDERED_MAP_DECLARATION_5(key, type, hash, equal, name)\
    typedef ::hpx::server::partition_unordered_map<key, type, hash, equal>    \
        BOOST_PP_CAT(partition_unordered_map, __LINE__);                      \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::get_value_action,    \
        BOOST_PP_CAT(__unordered_map_get_value_action_, name));               \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::get_values_action,   \
        BOOST_PP_CAT(__unordered_map_get_values_action_, name));              \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::set_value_action,    \
        BOOST_PP_CAT(__unordered_map_set_value_action_, name));               \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::set_values_action,   \
        BOOST_PP_CAT(__unordered_map_set_values_action_, name));              \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::size_action,         \
        BOOST_PP_CAT(__unordered_map_size_action_, name));                    \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::erase_action,        \
        BOOST_PP_CAT(__unordered_map_erase_action_, name));                   \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::get_copied_data_action,\
        BOOST_PP_CAT(__unordered_map_get_copied_data_action_, name));         \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::set_copied_data_action,\
        BOOST_PP_CAT(__unordered_map_set_copied_data_action_, name));         \
    typedef std::plus<std::size_t>                                            \
        BOOST_PP_CAT(partition_unordered_map_size_reduceop, __LINE__);        \
    typedef BOOST_PP_CAT(partition_unordered_map, __LINE__)::size_action      \
        BOOST_PP_CAT(BOOST_PP_CAT(partition_unordered_map, size_action),      \
            __LINE__);                                                        \
    HPX_REGISTER_REDUCE_ACTION_DECLARATION(                                   \
        BOOST_PP_CAT(BOOST_PP_CAT(partition_unordered_map, size_action),      \
            __LINE__),                                                        \
        BOOST_PP_CAT(partition_unordered_map_size_reduceop, __LINE__));       \
/**/

#define HPX_REGISTER_UNORDERED_MAP(...)                                       \
    HPX_REGISTER_UNORDERED_MAP_(__VA_ARGS__)                                  \
/**/
#define HPX_REGISTER_UNORDERED_MAP_(...)                                      \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_UNORDERED_MAP_, HPX_UTIL_PP_NARG(__VA_ARGS__)            \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_UNORDERED_MAP_2(key, type)                               \
    HPX_REGISTER_UNORDERED_MAP_5(key, type, std::hash<key>,                   \
        std::equal_to<key>, type)                                             \
/**/
#define HPX_REGISTER_UNORDERED_MAP_3(key, type, hash)                         \
    HPX_REGISTER_UNORDERED_MAP_5(key, type, hash,                             \
        std::equal_to<key>, type)                                             \
/**/
#define HPX_REGISTER_UNORDERED_MAP_4(key, type, hash, equal)                  \
    HPX_REGISTER_UNORDERED_MAP_5(key, type, hash, equal, type)                \
/**/

#define HPX_REGISTER_UNORDERED_MAP_5(key, type, hash, equal, name)            \
    typedef ::hpx::server::partition_unordered_map<key, type, hash, equal>    \
        BOOST_PP_CAT(partition_unordered_map, __LINE__);                      \
    HPX_REGISTER_ACTION(                                                      \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::get_value_action,    \
        BOOST_PP_CAT(__unordered_map_get_value_action_, name));               \
    HPX_REGISTER_ACTION(                                                      \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::get_values_action,   \
        BOOST_PP_CAT(__unordered_map_get_values_action_, name));              \
    HPX_REGISTER_ACTION(                                                      \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::set_value_action,    \
        BOOST_PP_CAT(__unordered_map_set_value_action_, name));               \
    HPX_REGISTER_ACTION(                                                      \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::set_values_action,   \
        BOOST_PP_CAT(__unordered_map_set_values_action_, name));              \
    HPX_REGISTER_ACTION(                                                      \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::size_action,         \
        BOOST_PP_CAT(__unordered_map_size_action_, name));                    \
    HPX_REGISTER_ACTION(                                                      \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::erase_action,        \
        BOOST_PP_CAT(__unordered_map_erase_action_, name));                   \
    HPX_REGISTER_ACTION(                                                      \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::get_copied_data_action,\
        BOOST_PP_CAT(__unordered_map_get_copied_data_action_, name));         \
    HPX_REGISTER_ACTION(                                                      \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)::set_copied_data_action,\
        BOOST_PP_CAT(__unordered_map_set_copied_data_action_, name));         \
    typedef std::plus<std::size_t>                                            \
        BOOST_PP_CAT(partition_unordered_map_size_reduceop, __LINE__);        \
    typedef BOOST_PP_CAT(partition_unordered_map, __LINE__)::size_action      \
        BOOST_PP_CAT(BOOST_PP_CAT(partition_unordered_map, size_action),      \
            __LINE__);                                                        \
    HPX_REGISTER_REDUCE_ACTION(                                               \
        BOOST_PP_CAT(BOOST_PP_CAT(partition_unordered_map, size_action),      \
            __LINE__),                                                        \
        BOOST_PP_CAT(partition_unordered_map_size_reduceop, __LINE__));       \
    typedef ::hpx::components::simple_component<                              \
        BOOST_PP_CAT(partition_unordered_map, __LINE__)                       \
    > BOOST_PP_CAT(__unordered_map_, name);                                   \
    HPX_REGISTER_COMPONENT(BOOST_PP_CAT(__unordered_map_, name))              \
/**/

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    template <typename Key, typename T, typename Hash = std::hash<Key>,
        typename KeyEqual = std::equal_to<Key> >
    class partition_unordered_map
      : public components::client_base<
            partition_unordered_map<Key, T, Hash, KeyEqual>,
            server::partition_unordered_map<Key, T, Hash, KeyEqual>
        >
    {
    private:
        typedef hpx::server::partition_unordered_map<Key, T, Hash, KeyEqual>
            server_type;
        typedef hpx::components::client_base<
                partition_unordered_map<Key, T, Hash, KeyEqual>,
                server::partition_unordered_map<Key, T, Hash, KeyEqual>
            > base_type;

    public:
        partition_unordered_map() {}

        partition_unordered_map(id_type const& gid)
          : base_type(gid)
        {}

        partition_unordered_map(hpx::shared_future<id_type> const& gid)
          : base_type(gid)
        {}

        // Return the pinned pointer to the underlying component
        std::shared_ptr<server::partition_unordered_map<Key, T, Hash, KeyEqual> >
        get_ptr() const
        {
            error_code ec(lightweight);
            return hpx::get_ptr<server_type>(this->get_id()).get(ec);
        }

        ///////////////////////////////////////////////////////////////////////
        //  Capacity related API's in partition_unordered_map client class

        /// Asynchronously return the size of the partition_unordered_map component.
        ///
        /// \return This returns size as the hpx::future of type size_type
        ///
        future<std::size_t> size_async() const
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::size_action>(this->get_id());
        }

        /// Return the size of the partition_unordered_map component.
        ///
        /// \return This returns size as the hpx::future of type size_type
        ///
        std::size_t size() const
        {
            return size_async().get();
        }

        //  Element Access API's in Client class

        /// Returns the value at position \a pos in the partition_unordered_map
        /// component.
        ///
        /// \param pos  Position of the element in the partition_unordered_map
        ///
        /// \return Returns the value of the element at position represented
        ///         by \a pos
        ///
        T get_value_sync(Key const& pos, bool erase) const
        {
            return get_value(pos, erase).get();
        }

        /// Return the element at the position \a pos in the
        /// partition_unordered_map container.
        ///
        /// \param pos Position of the element in the partition_unordered_map
        ///
        /// \return This returns the value as the hpx::future
        ///
        future<T> get_value(Key const& pos, bool erase) const
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::get_value_action>(
                this->get_id(), pos, erase);
        }

        /// Returns the value at position \a pos in the partition_unordered_map
        /// component.
        ///
        /// \param pos  Position of the element in the partition_unordered_map
        ///
        /// \return Returns the value of the element at position represented
        ///         by \a pos
        ///
        std::vector<T> get_values_sync(std::vector<Key> const& keys) const
        {
            return get_values(keys).get();
        }

        /// Return the element at the position \a pos in the
        /// partition_unordered_map container.
        ///
        /// \param pos Position of the element in the partition_unordered_map
        ///
        /// \return This returns the value as the hpx::future
        ///
        future<std::vector<T> > get_values(std::vector<Key> const& keys) const
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::get_values_action>(
                this->get_id(), keys);
        }

        /// Copy the value of \a val in the element at position
        /// \a pos in the partition_unordered_map container.
        ///
        /// \param pos   Position of the element in the partition_unordered_map
        /// \param val   The value to be copied
        ///
        template <typename T_>
        void set_value_sync(Key const& pos, T_ && val)
        {
            set_value(pos, std::forward<T_>(val)).get();
        }

        /// Copy the value of \a val in the element at position
        /// \a pos in the partition_unordered_map component.
        ///
        /// \param pos  Position of the element in the partition_unordered_map
        /// \param val  Value to be copied
        ///
        /// \return This returns the hpx::future of type void
        ///
        template <typename T_>
        future<void> set_value(Key const& pos, T_ && val)
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::set_value_action>(
                this->get_id(), pos, std::forward<T_>(val));
        }

        /// Copy the value of \a val in the element at position
        /// \a pos in the partition_unordered_map container.
        ///
        /// \param pos   Position of the element in the partition_unordered_map
        /// \param val   The value to be copied
        ///
        void set_values_sync(std::vector<Key> const& keys,
            std::vector<T> const& vals)
        {
            set_values(keys, vals).get();
        }

        /// Copy the value of \a val in the element at position
        /// \a pos in the partition_unordered_map component.
        ///
        /// \param pos  Position of the element in the partition_unordered_map
        /// \param val  Value to be copied
        ///
        /// \return This returns the hpx::future of type void
        ///
        future<void> set_values(std::vector<Key> const& keys,
            std::vector<T> const& vals)
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::set_values_action>(
                this->get_id(), keys, vals);
        }

        /// Erase all values with the given key from the partition_unordered_map
        /// container.
        ///
        /// \param key   Key of the element in the partition_unordered_map
        ///
        /// \return Returns the number of elements erased
        ///
        std::size_t erase_sync(Key const& key)
        {
            return erase(key).get();
        }

        /// Erase all values with the given key from the partition_unordered_map
        /// container.
        ///
        /// \param key  Key of the element in the partition_unordered_map
        ///
        /// \return This returns the hpx::future containing the number of
        ///         elements erased
        ///
        future<std::size_t> erase(Key const& key)
        {
            HPX_ASSERT(this->get_id());
            return hpx::async<typename server_type::erase_action>(
                this->get_id(), key);
        }

        /// Get/set all the data of this partition
        future<typename server_type::data_type> get_data() const
        {
            typedef typename server_type::get_copied_data_action action_type;
            return async<action_type>(this->get_id());
        }

        future<void> set_data(typename server_type::data_type && d)
        {
            typedef typename server_type::set_copied_data_action action_type;
            return async<action_type>(this->get_id(), std::move(d));
        }
    };
}

#endif
