// Copyright (c) 2019 Weile Wei
// Copyright (c) 2019 Maxwell Reeser
// Copyright (c) 2019 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/distributed_object.hpp

#ifndef HPX_LCOS_DISTRIBUTED_OBJECT_HPP
#define HPX_LCOS_DISTRIBUTED_OBJECT_HPP

#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/barrier.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/serialization/unordered_map.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/// \cond NOINTERNAL
namespace hpx { namespace lcos { namespace server {
    template <typename T>
    class distributed_object_part
      : public hpx::components::locking_hook<
            hpx::components::component_base<distributed_object_part<T>>>
    {
    public:
        typedef T data_type;
        distributed_object_part() {}

        distributed_object_part(data_type const& data)
          : data_(data)
        {
        }

        distributed_object_part(data_type&& data)
          : data_(std::move(data))
        {
        }

        data_type& operator*()
        {
            return data_;
        }

        data_type const& operator*() const
        {
            return data_;
        }

        data_type const* operator->() const
        {
            return &data_;
        }

        data_type* operator->()
        {
            return &data_;
        }

        data_type fetch() const
        {
            return data_;
        }

        HPX_DEFINE_COMPONENT_ACTION(distributed_object_part, fetch);

    private:
        data_type data_;
    };

    template <typename T>
    class distributed_object_part<T&>
      : public hpx::components::locking_hook<
            hpx::components::component_base<distributed_object_part<T&>>>
    {
    public:
        typedef T& data_type;
        distributed_object_part() {}

        distributed_object_part(data_type data)
          : data_(data)
        {
        }

        data_type operator*()
        {
            return data_;
        }

        data_type operator*() const
        {
            return data_;
        }

        T const* operator->() const
        {
            return data_;
        }

        T* operator->()
        {
            return data_;
        }

        T fetch() const
        {
            return data_;
        }

        HPX_DEFINE_COMPONENT_ACTION(distributed_object_part, fetch);

    private:
        data_type data_;
    };
}}}

#define REGISTER_DISTRIBUTED_OBJECT_PART_DECLARATION(type)                     \
    HPX_REGISTER_ACTION_DECLARATION(                                           \
        hpx::lcos::server::distributed_object_part<type>::fetch_action,        \
        HPX_PP_CAT(__distributed_object_part_fetch_action_, type));

/**/

#define REGISTER_DISTRIBUTED_OBJECT_PART(type)                                 \
    HPX_REGISTER_ACTION(                                                       \
        hpx::lcos::server::distributed_object_part<type>::fetch_action,        \
        HPX_PP_CAT(__distributed_object_part_fetch_action_, type));            \
    typedef ::hpx::components::component<                                      \
        hpx::lcos::server::distributed_object_part<type>>                      \
        HPX_PP_CAT(__distributed_object_part_, type);                          \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(__distributed_object_part_, type))       \
    /**/

namespace hpx { namespace lcos {
    enum class construction_type
    {
        Meta_Object,
        All_to_All
    };
}}

namespace hpx { namespace lcos {
    class meta_object_server
      : public hpx::components::locking_hook<
            hpx::components::component_base<meta_object_server>>
    {
    public:
        meta_object_server()
          : b(hpx::find_all_localities().size())
        {
            servers.resize(hpx::find_all_localities().size());
        }

        meta_object_server(std::size_t num_locs, std::size_t root)
          : b(num_locs)
        {
            servers_ = std::unordered_map<std::size_t, hpx::id_type>();
            retrieved = false;
        }

        std::unordered_map<std::size_t, hpx::id_type> get_server_list()
        {
            return servers_;
        }

        std::unordered_map<std::size_t, hpx::id_type> registration(
            std::size_t source_loc, hpx::id_type id)
        {
            {
                servers_[source_loc] = id;
            }
            b.wait();
            return servers_;
        }

        HPX_DEFINE_COMPONENT_ACTION(meta_object_server, get_server_list);
        HPX_DEFINE_COMPONENT_ACTION(meta_object_server, registration);

    private:
        hpx::lcos::local::barrier b;
        std::vector<hpx::id_type> servers;
        bool retrieved;
        std::unordered_map<std::size_t, hpx::id_type> servers_;
        std::vector<hpx::id_type> servers__;
    };
}}
typedef hpx::lcos::meta_object_server::get_server_list_action get_list_action;
HPX_REGISTER_ACTION_DECLARATION(get_list_action, get_server_list_mo_action);
typedef hpx::lcos::meta_object_server::registration_action
    register_with_meta_action;
HPX_REGISTER_ACTION_DECLARATION(register_with_meta_action, register_mo_action);

// Meta_object front end, decides whether it is the root locality, and thus
// whether to register with the root locality's meta object only or to register
// itself as the root locality's meta object as well
namespace hpx { namespace lcos {
    class meta_object
      : hpx::components::client_base<meta_object, meta_object_server>
    {
    public:
        typedef hpx::components::client_base<meta_object, meta_object_server>
            base_type;
        meta_object(
            std::string basename, std::size_t num_locs, std::size_t root)
          : base_type(
                hpx::new_<meta_object_server>(hpx::find_here(), num_locs, root))
        {
            if (hpx::get_locality_id() == root)
            {
                hpx::register_with_basename(
                    basename, this->get_id(), hpx::get_locality_id());
            }
            meta_object_0 = hpx::find_from_basename(basename, root).get();
        }

        std::unordered_map<std::size_t, hpx::id_type> get_server_list()
        {
            return hpx::async(get_list_action(), meta_object_0).get();
        }

        std::unordered_map<std::size_t, hpx::id_type> registration(
            hpx::id_type id)
        {
            hpx::future<std::unordered_map<std::size_t, hpx::id_type>> ret =
                hpx::async(register_with_meta_action(), meta_object_0,
                    hpx::get_locality_id(), id);
            std::unordered_map<std::size_t, hpx::id_type> ret_ = ret.get();
            return ret_;
        }

    private:
        hpx::id_type meta_object_0;
    };
}}
/// \endcond
// The front end for the distributed_object itself. Essentially wraps actions for
// the server, and stores information locally about the localities/servers
// that it needs to know about
namespace hpx { namespace lcos {
    /// The distributed_object is a single logical object partitioned over a set of
    /// localities/nodes/machines, where every locality shares the same global
    /// name locality for the distributed object (i.e. a universal name), but
    /// owns its local value. In other words, local data of the distributed
    /// object can be different, but they share access to one another's data
    /// globally.
    template <typename T, construction_type C = construction_type::All_to_All>
    class distributed_object
      : hpx::components::client_base<distributed_object<T>,
            server::distributed_object_part<T>>
    {
        typedef hpx::components::client_base<distributed_object<T>,
            server::distributed_object_part<T>>
            base_type;

        typedef
            typename server::distributed_object_part<T>::data_type data_type;

    private:
        template <typename Arg>
        static hpx::future<hpx::id_type> create_server(Arg&& value)
        {
            return hpx::local_new<server::distributed_object_part<T>>(
                std::forward<Arg>(value));
        }

    public:
        /// Creates a distributed_object in every locality
        ///
        /// A distributed_object \a base_name is created through default constructor.
        distributed_object() = default;

        /// Creates a distributed_object in every locality with a given base_name string,
        /// data, and a type and construction_type in the template parameters
        ///
        /// \param construction_type The construction_type in the template parameters
        /// accepts either Meta_Object, and it is set to All_to_All by defalut
        /// The Meta_Object option provides meta object registration in the root
        /// locality and meta object is essentailly a table that can find the
        /// instances of distributed_object in all localities. The All_to_All option only
        /// locally holds the client and server of the distributed_object.
        /// \param base_name The name of the distributed_object, which should be a unique
        /// string across the localities
        /// \param data The data of the type T of the distributed_object
        distributed_object(std::string base, data_type const& data,
            std::vector<size_t> sub_localities = all_localities())
          : base_type(create_server(data))
          , base_(base)
          , sub_localities_(std::move(sub_localities))
        {
            HPX_ASSERT(C == construction_type::All_to_All ||
                C == construction_type::Meta_Object);
            HPX_ASSERT(sub_localities_.size() > 0);
            ensure_ptr();
            std::sort(sub_localities_.begin(), sub_localities_.end());

            if (C == construction_type::Meta_Object)
            {
                meta_object mo(
                    base, sub_localities_.size(), sub_localities_[0]);
                locs = mo.registration(this->get_id());
                basename_registration_helper(base, sub_localities_.size());
            }
            else
            {
                if (std::find(sub_localities_.begin(), sub_localities_.end(),
                        static_cast<size_t>(hpx::get_locality_id())) !=
                    sub_localities_.end())
                {
                    basename_registration_helper(base, sub_localities_.size());
                    hpx::lcos::barrier barrier("wait_for_all_constructors",
                        hpx::find_all_localities().size(),
                        hpx::get_locality_id());
                    barrier.wait();
                }
                else
                {
                    HPX_THROW_EXCEPTION(hpx::no_success, "constructor error",
                        "distributed object is not valid within the given "
                        "sub-locality");
                }
            }
        }
        /// Creates a distributed_object in every locality with a given base_name string,
        /// data. The construction_type in the template parameter is set to
        /// All_to_All option by default.
        ///
        /// \param base_name The name of the distributed_object, which should be a unique
        /// string across the localities
        /// \param data The data of the type T of the distributed_object
        distributed_object(std::string base, data_type&& data,
            std::vector<size_t> sub_localities = all_localities())
          : base_type(create_server(std::move(data)))
          , base_(base)
          , sub_localities_(std::move(sub_localities))
        {
            ensure_ptr();
            basename_registration_helper(
                base, hpx::find_all_localities().size());
        }
        /// \cond NOINTERNAL
        /// generate boilerplate code for the client
        distributed_object(hpx::future<hpx::id_type>&& id)
          : base_type(std::move(id))
        {
        }
        /// generate boilerplate code for the client
        distributed_object(hpx::id_type&& id)
          : base_type(std::move(id))
        {
        }
        /// \endcond

        /// Access the calling locality's value instance for this distributed_object
        data_type const& operator*() const
        {
            HPX_ASSERT(this->get_id());
            return **ptr;
        }

        /// Access the calling locality's value instance for this distributed_object
        data_type& operator*()
        {
            HPX_ASSERT(this->get_id());
            return **ptr;
        }

        /// Access the calling locality's value instance for this distributed_object
        data_type const* operator->() const
        {
            HPX_ASSERT(this->get_id());
            return &**ptr;
        }

        /// Access the calling locality's value instance for this distributed_object
        data_type* operator->()
        {
            HPX_ASSERT(this->get_id());
            return &**ptr;
        }

        /// Asynchronously returns a future of a copy of the instance of this
        /// distributed_object associated with the given locality index. The locality
        /// index must be a valid locality ID with this distributed_object.
        // TODO: write exception description
        hpx::future<data_type> fetch(int idx)
        {
            /// \cond NOINTERNAL
            if (std::find(sub_localities_.begin(), sub_localities_.end(),
                    static_cast<size_t>(hpx::get_locality_id())) !=
                sub_localities_.end())
            {
                HPX_ASSERT(this->get_id());
                hpx::id_type lookup = get_basename_helper(idx);
                typedef
                    typename server::distributed_object_part<T>::fetch_action
                        action_type;
                return hpx::async<action_type>(lookup);
            }
            else
            {
                HPX_THROW_EXCEPTION(hpx::no_success,
                    "fetch error",
                    "distributed object is not valid within the given "
                    "locality");
            }
            /// \endcond
        }
        /// \cond NOINTERNAL
        /// force compilation error if serialization of client occurs
        template <typename Archive, typename Type>
        HPX_FORCEINLINE void serialize(
            Archive& ar, base_type& f, unsigned version) = delete;

    private:
        mutable std::shared_ptr<server::distributed_object_part<T>> ptr;
        std::string base_;
        std::vector<size_t> sub_localities_;
        void ensure_ptr() const
        {
            ptr = hpx::get_ptr<server::distributed_object_part<T>>(
                hpx::launch::sync, this->get_id());
        }

        // make sure sub_localities_ is initialized ranging from 0 to
        // num of all localities when the constructor is called within
        // the context that all localities are provided so that fetch
        // function can identify the target locality that can be found
        // from the sub_localities
        static std::vector<size_t> all_localities()
        {
            std::vector<size_t> all_localities_tmp;
            all_localities_tmp.resize(hpx::find_all_localities().size());
            std::iota(all_localities_tmp.begin(), all_localities_tmp.end(), 0);
            return all_localities_tmp;
        }

    private:
        std::vector<hpx::id_type> basename_list;
        std::unordered_map<std::size_t, hpx::id_type> locs;
        hpx::id_type get_basename_helper(int idx)
        {
            if (!locs[idx])
            {
                locs[idx] =
                    hpx::find_from_basename(base_ + std::to_string(idx), idx)
                        .get();
            }
            return locs[idx];
        }
        void basename_registration_helper(
            std::string base, size_t basename_list_size)
        {
            hpx::register_with_basename(
                base + std::to_string(hpx::get_locality_id()), this->get_id(),
                hpx::get_locality_id());
            basename_list.resize(basename_list_size);
        }
        /// \endcond
    };

    /// The distributed_object is a single logical object partitioned over a set of
    /// localities/nodes/machines, where every locality shares the same global
    /// name locality for the distributed object (i.e. a universal name), but
    /// owns its local value. In other words, local data of the distributed
    /// object can be different, but they share access to one another's data
    /// globally.
    template <typename T, construction_type C>
    class distributed_object<T&, C>
      : hpx::components::client_base<distributed_object<T&>,
            server::distributed_object_part<T&>>
    {
        /// \cond NOINTERNAL
        typedef hpx::components::client_base<distributed_object<T&>,
            server::distributed_object_part<T&>>
            base_type;

        typedef
            typename server::distributed_object_part<T&>::data_type data_type;

    private:
        template <typename Arg>
        static hpx::future<hpx::id_type> create_server(Arg& value)
        {
            return hpx::local_new<server::distributed_object_part<T&>>(value);
        }
        /// \endcond
    public:
        /// Creates a distributed_object in every locality
        ///
        /// A distributed_object \a base_name is created through default constructor.
        distributed_object() = default;

        /// Creates a distributed_object in every locality with a given base_name string,
        /// data, and a construction_type. This constructor of the distributed_object
        /// wraps an existing local instance and thus is internally referring to
        /// the local instance.
        ///
        /// \param construction_type The construction_type in the template parameters
        /// accepts either Meta_Object, and it is set to All_to_All by defalut
        /// The Meta_Object option provides meta object registration in the root
        /// locality and meta object is essentailly a table that can find the
        /// instances of distributed_object in all localities. The All_to_All option only
        /// locally holds the client and server of the distributed_object.
        /// \param base_name The name of the distributed_object, which should be a unique
        /// string across the localities
        /// \param data The data of the type T of the distributed_object
        /*       distributed_object(std::string base, data_type data)
          : base_type(create_server(data))
          , base_(base)
        {
            HPX_ASSERT(C == construction_type::All_to_All ||
                C == construction_type::Meta_Object);
            ensure_ptr();
            size_t localities = hpx::find_all_localities().size();
            init_sub_localities();
            if (C == construction_type::Meta_Object)
            {
                meta_object mo(base, localities, 0);
                locs = mo.registration(this->get_id());
                basename_registration_helper(base, localities);
            }
            else
            {
                basename_registration_helper(base, localities);
            }
        }*/

        // TODO: Doxgen doc
        distributed_object(std::string base, data_type data,
            std::vector<size_t> sub_localities = all_localities())
          : base_type(create_server(data))
          , base_(base)
          , sub_localities_(std::move(sub_localities))
        {
            HPX_ASSERT(C == construction_type::All_to_All ||
                C == construction_type::Meta_Object);
            HPX_ASSERT(sub_localities_.size() > 0);
            ensure_ptr();
            std::sort(sub_localities_.begin(), sub_localities_.end());

            if (C == construction_type::Meta_Object)
            {
                meta_object mo(
                    base, sub_localities_.size(), sub_localities_[0]);
                locs = mo.registration(this->get_id());
                basename_registration_helper(base, sub_localities_.size());
            }
            else
            {
                if (std::find(sub_localities_.begin(), sub_localities_.end(),
                        static_cast<size_t>(hpx::get_locality_id())) !=
                    sub_localities_.end())
                {
                    basename_registration_helper(base, sub_localities_.size());
                    hpx::lcos::barrier barrier("wait_for_all_constructors",
                        hpx::find_all_localities().size(),
                        hpx::get_locality_id());
                    barrier.wait();
                }
                else
                {
                    HPX_THROW_EXCEPTION(hpx::no_success, "constructor error",
                        "distributed object is not valid within the given "
                        "sub-locality");
                }
            }
        }

        /// \cond NOINTERNAL
        /// generate boilerplate code for the client
        distributed_object(hpx::future<hpx::id_type>&& id)
          : base_type(std::move(id))
        {
        }
        /// generate boilerplate code for the client
        distributed_object(hpx::id_type&& id)
          : base_type(std::move(id))
        {
        }
        /// \endcond

        /// Access the calling locality's value instance for this distributed_object
        data_type const operator*() const
        {
            HPX_ASSERT(this->get_id());
            return **ptr;
        }

        /// Access the calling locality's value instance for this distributed_object
        data_type operator*()
        {
            HPX_ASSERT(this->get_id());
            return **ptr;
        }

        /// Access the calling locality's value instance for this distributed_object
        T const* operator->() const
        {
            HPX_ASSERT(this->get_id());
            return &**ptr;
        }

        /// Access the calling locality's value instance for this distributed_object
        T* operator->()
        {
            HPX_ASSERT(this->get_id());
            return &**ptr;
        }

        /// Asynchronously returns a future of a copy of the instance of this
        /// distributed_object associated with the given locality index. The locality
        /// index must be a valid locality ID with this distributed_object.
        // TODO: write exception description
        hpx::future<T> fetch(int idx)
        {
            if (std::find(sub_localities_.begin(), sub_localities_.end(),
                    static_cast<size_t>(hpx::get_locality_id())) !=
                sub_localities_.end())
            {
                HPX_ASSERT(this->get_id());
                hpx::id_type lookup = get_basename_helper(idx);
                typedef typename server::distributed_object_part<
                    T&>::fetch_ref_action action_type;
                return hpx::async<action_type>(lookup);
            }
            else
            {
                HPX_THROW_EXCEPTION(hpx::no_success, "fetch error",
                    "distributed object is not valid within the given "
                    "locality");
            }
        }
        /// \cond NOINTERNAL
        /// force compilation error if serialization of client occurs
        template <typename Archive, typename Type>
        HPX_FORCEINLINE void serialize(
            Archive& ar, base_type& f, unsigned version) = delete;

    private:
        mutable std::shared_ptr<server::distributed_object_part<T&>> ptr;
        std::string base_;
        mutable std::vector<size_t> sub_localities_;

        // make sure sub_localities_ is initialized ranging from 0 to
        // num of all localities when the constructor is called within
        // the context that all localities are provided so that fetch
        // function can identify the target locality that can be found
        // from the sub_localities
        static std::vector<size_t> all_localities()
        {
            std::vector<size_t> all_localities_tmp;
            all_localities_tmp.resize(hpx::find_all_localities().size());
            std::iota(all_localities_tmp.begin(), all_localities_tmp.end(), 0);
            return all_localities_tmp;
        }

        void ensure_ptr() const
        {
            ptr = hpx::get_ptr<server::distributed_object_part<T&>>(
                hpx::launch::sync, this->get_id());
        }

    private:
        std::vector<hpx::id_type> basename_list;
        std::unordered_map<std::size_t, hpx::id_type> locs;
        hpx::id_type get_basename_helper(int idx)
        {
            if (!locs[idx])
            {
                locs[idx] =
                    hpx::find_from_basename(base_ + std::to_string(idx), idx)
                        .get();
            }
            return locs[idx];
        }
        void basename_registration_helper(
            std::string base, size_t basename_list_size)
        {
            hpx::register_with_basename(
                base + std::to_string(hpx::get_locality_id()), this->get_id(),
                hpx::get_locality_id());
            basename_list.resize(basename_list_size);
        }
        /// \endcond
    };
}}
#endif /*HPX_LCOS_DISTRIBUTED_OBJECT_HPP*/
