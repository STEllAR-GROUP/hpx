// Copyright (c) 2019 Weile Wei
// Copyright (c) 2019 Maxwell Reeser
// Copyright (c) 2019 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/dist_object.hpp

#ifndef HPX_LCOS_DIST_OBJECT_HPP
#define HPX_LCOS_DIST_OBJECT_HPP

#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/barrier.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/detail/pp/cat.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
/// \cond NOINTERNAL
namespace hpx { namespace lcos { namespace dist_object { namespace server {
    template <typename T>
    class dist_object_part
      : public hpx::components::locking_hook<
            hpx::components::component_base<dist_object_part<T>>>
    {
    public:
        typedef T data_type;
        dist_object_part() {}

        dist_object_part(data_type const& data)
          : data_(data)
        {
        }

        dist_object_part(data_type&& data)
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

        HPX_DEFINE_COMPONENT_ACTION(dist_object_part, fetch);

    private:
        data_type data_;
    };

    template <typename T>
    class dist_object_part<T&>
      : public hpx::components::locking_hook<
            hpx::components::component_base<dist_object_part<T&>>>
    {
    public:
        typedef T& data_type;
        dist_object_part() {}

        dist_object_part(data_type data)
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

        HPX_DEFINE_COMPONENT_ACTION(dist_object_part, fetch);

    private:
        data_type data_;
    };
}}}}

#define REGISTER_DIST_OBJECT_PART_DECLARATION(type)                            \
    HPX_REGISTER_ACTION_DECLARATION(                                           \
        hpx::lcos::dist_object::server::dist_object_part<type>::fetch_action,  \
        HPX_PP_CAT(__dist_object_part_fetch_action_, type));

/**/

#define REGISTER_DIST_OBJECT_PART(type)                                        \
    HPX_REGISTER_ACTION(                                                       \
        hpx::lcos::dist_object::server::dist_object_part<type>::fetch_action,  \
        HPX_PP_CAT(__dist_object_part_fetch_action_, type));                   \
    typedef ::hpx::components::component<                                      \
        hpx::lcos::dist_object::server::dist_object_part<type>>                \
        HPX_PP_CAT(__dist_object_part_, type);                                 \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(__dist_object_part_, type))              \
    /**/

namespace hpx { namespace lcos { namespace dist_object {
    enum class construction_type
    {
        Meta_Object,
        All_to_All
    };
}}}

namespace hpx { namespace lcos { namespace dist_object {
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

        std::vector<hpx::id_type> get_server_list()
        {
            if (servers.size() == hpx::find_all_localities().size())
            {
                num_sent++;
                return servers;
            }
            std::vector<hpx::id_type> empty;
            empty.resize(0);
            return empty;
        }

        std::vector<hpx::id_type> registration(
            std::size_t source_loc, hpx::id_type id)
        {
            {
                // TODO: duplicated lock: the namespace has lock guard already
                std::lock_guard<hpx::lcos::local::spinlock> l(lk);
                servers[source_loc] = id;
            }
            b.wait();
            return servers;
        }

        HPX_DEFINE_COMPONENT_ACTION(meta_object_server, get_server_list);
        HPX_DEFINE_COMPONENT_ACTION(meta_object_server, registration);

    private:
        int num_sent;
        hpx::lcos::local::barrier b;
        hpx::lcos::local::spinlock lk;
        std::vector<hpx::id_type> servers;
    };
}}}
typedef hpx::lcos::dist_object::meta_object_server::get_server_list_action
    get_list_action;
HPX_REGISTER_ACTION_DECLARATION(get_list_action, get_server_list_mo_action);
typedef hpx::lcos::dist_object::meta_object_server::registration_action
    register_with_meta_action;
HPX_REGISTER_ACTION_DECLARATION(register_with_meta_action, register_mo_action);

// Meta_object front end, decides whether it is the root locality, and thus
// whether to register with the root locality's meta object only or to register
// itself as the root locality's meta object as well
namespace hpx { namespace lcos { namespace dist_object {
    class meta_object
      : hpx::components::client_base<meta_object, meta_object_server>
    {
    public:
        typedef hpx::components::client_base<meta_object, meta_object_server>
            base_type;

        meta_object(std::string basename)
          : base_type(hpx::local_new<meta_object_server>())
        {
            if (hpx::get_locality_id() == 0)
            {
                hpx::register_with_basename(
                    basename, this->get_id(), hpx::get_locality_id());
            }
            meta_object_0 = hpx::find_from_basename(basename, 0).get();
        }

        hpx::future<std::vector<hpx::id_type>> get_server_list()
        {
            return hpx::async(get_list_action(), meta_object_0);
        }

        std::vector<hpx::id_type> registration(hpx::id_type id)
        {
            return hpx::async(register_with_meta_action(), meta_object_0,
                hpx::get_locality_id(), id)
                .get();
        }

    private:
        hpx::id_type meta_object_0;
    };
}}}
/// \endcond
// The front end for the dist_object itself. Essentially wraps actions for
// the server, and stores information locally about the localities/servers
// that it needs to know about
namespace hpx { namespace lcos { namespace dist_object {
    /// The dist_object is a single logical object partitioned over a set of
    /// localities/nodes/machines, where every locality shares the same global
    /// name locality for the distributed object (i.e. a universal name), but
    /// owns its local value. In other words, local data of the distributed
    /// object can be different, but they share access to one another's data
    /// globally.
    template <typename T, construction_type C = construction_type::All_to_All>
    class dist_object
      : hpx::components::client_base<dist_object<T>,
            server::dist_object_part<T>>
    {
        typedef hpx::components::client_base<dist_object<T>,
            server::dist_object_part<T>>
            base_type;

        typedef typename server::dist_object_part<T>::data_type data_type;

    private:
        template <typename Arg>
        static hpx::future<hpx::id_type> create_server(Arg&& value)
        {
            return hpx::local_new<server::dist_object_part<T>>(
                std::forward<Arg>(value));
        }

    public:
        /// Creates a dist_object in every locality
        ///
        /// A dist_object \a base_name is created through default constructor.
        dist_object() {}

        /// Creates a dist_object in every locality with a given base_name string,
        /// data, and a construction_type
        ///
        /// \param base_name The name of the dist_object, which should be a unique
        /// string across the localities
        /// \param data The data of the type T of the dist_object
        /// \param construction_type The construction_type accepts either Meta_Object
        /// or All_to_All option. The Meta_Object option provides meta object
        /// registration in the root locality and meta object is essentailly a table
        /// that can find the instances of dist_object in all localities. The
        /// All_to_All option only locally holds the client and server of the
        /// dist_object.
        dist_object(
            std::string base, data_type const& data)
          : base_type(create_server(data))
          , base_(base)
        {
            HPX_ASSERT(C == construction_type::All_to_All ||
                C == construction_type::Meta_Object);
            if (C == construction_type::Meta_Object)
            {
                meta_object mo(base);
                basename_list = mo.registration(this->get_id());
                basename_registration_helper(base);
            }
            else
            {
                basename_registration_helper(base);
            }
        }

        /// Creates a dist_object in every locality with a given base_name string,
        /// data. The construction_type is not provided in this constructor and is
        /// set to All_to_All option by default.
        ///
        /// \param base_name The name of the dist_object, which should be a unique
        /// string across the localities
        /// \param data The data of the type T of the dist_object
        dist_object(std::string base, data_type&& data)
          : base_type(create_server(std::move(data)))
          , base_(base)
        {
            basename_registration_helper(base);
        }
        /// \cond NOINTERNAL
        dist_object(hpx::future<hpx::id_type>&& id)
          : base_type(std::move(id))
        {
        }

        dist_object(hpx::id_type&& id)
          : base_type(std::move(id))
        {
        }
        /// \endcond

        /// Access the calling locality's value instance for this dist_object
        data_type const& operator*() const
        {
            HPX_ASSERT(this->get_id());
            ensure_ptr();
            return **ptr;
        }

        /// Access the calling locality's value instance for this dist_object
        data_type& operator*()
        {
            HPX_ASSERT(this->get_id());
            ensure_ptr();
            return **ptr;
        }

        /// Access the calling locality's value instance for this dist_object
        data_type const* operator->() const
        {
            HPX_ASSERT(this->get_id());
            ensure_ptr();
            return &**ptr;
        }

        /// Access the calling locality's value instance for this dist_object
        data_type* operator->()
        {
            HPX_ASSERT(this->get_id());
            ensure_ptr();
            return &**ptr;
        }

        /// Asynchronously returns a future of a copy of the instance of this
        /// dist_object associated with the given locality index. The locality
        /// index must be a valid locality ID with this dist_object.
        hpx::future<data_type> fetch(int idx)
        {
            /// \cond NOINTERNAL
            HPX_ASSERT(this->get_id());
            hpx::id_type lookup = get_basename_helper(idx);
            typedef
                typename server::dist_object_part<T>::fetch_action action_type;
            return hpx::async<action_type>(lookup);
            /// \endcond
        }
        /// \cond NOINTERNAL
    private:
        mutable std::shared_ptr<server::dist_object_part<T>> ptr;
        std::string base_;
        std::string base_unpacked;
        void ensure_ptr() const
        {
            if (!ptr)
            {
                ptr = hpx::get_ptr<server::dist_object_part<T>>(
                    hpx::launch::sync, this->get_id());
            }
        }

    private:
        std::vector<hpx::id_type> basename_list;
        hpx::id_type get_basename_helper(int idx)
        {
            if (!basename_list[idx])
            {
                basename_list[idx] =
                    hpx::find_from_basename(base_ + std::to_string(idx), idx)
                        .get();
            }
            return basename_list[idx];
        }
        void basename_registration_helper(std::string base)
        {
            base_unpacked = base + std::to_string(hpx::get_locality_id());
            hpx::register_with_basename(
                base + std::to_string(hpx::get_locality_id()), this->get_id());
            basename_list.resize(hpx::find_all_localities().size());
        }
        /// \endcond
    };

    /// The dist_object is a single logical object partitioned over a set of
    /// localities/nodes/machines, where every locality shares the same global
    /// name locality for the distributed object (i.e. a universal name), but
    /// owns its local value. In other words, local data of the distributed
    /// object can be different, but they share access to one another's data
    /// globally.
    template <typename T, construction_type C>
    class dist_object<T&, C>
      : hpx::components::client_base<dist_object<T&>,
            server::dist_object_part<T&>>
    {
        /// \cond NOINTERNAL
        typedef hpx::components::client_base<dist_object<T&>,
            server::dist_object_part<T&>>
            base_type;

        typedef typename server::dist_object_part<T&>::data_type data_type;

    private:
        template <typename Arg>
        static hpx::future<hpx::id_type> create_server(Arg& value)
        {
            return hpx::local_new<server::dist_object_part<T&>>(value);
        }
        /// \endcond
    public:
        /// Creates a dist_object in every locality
        ///
        /// A dist_object \a base_name is created through default constructor.
        dist_object() {}

        /// Creates a dist_object in every locality with a given base_name string,
        /// data, and a construction_type. This constructor of the dist_object
        /// wraps an existing local instance and thus is internally referring to
        /// the local instance.
        ///
        /// \param base_name The name of the dist_object, which should be a unique
        /// string across the localities
        /// \param data The data of the type T of the dist_object
        /// \param construction_type The construction_type accepts either Meta_Object
        /// or All_to_All option. The Meta_Object option provides meta object
        /// registration in the root locality and meta object is essentailly a table
        /// that can find the instances of dist_object in all localities. The
        /// All_to_All option only locally holds the client and server of the
        /// dist_object.
        dist_object(std::string base, data_type data)
          : base_type(create_server(data))
          , base_(base)
        {
            HPX_ASSERT(C == construction_type::All_to_All ||
                C == construction_type::Meta_Object);
            if (C == construction_type::Meta_Object)
            {
                meta_object mo(base);
                basename_list = mo.registration(this->get_id());
                basename_registration_helper(base);
            }
            else
            {
                basename_registration_helper(base);
            }
        }

        /// \cond NOINTERNAL
        dist_object(hpx::future<hpx::id_type>&& id)
          : base_type(std::move(id))
        {
        }

        dist_object(hpx::id_type&& id)
          : base_type(std::move(id))
        {
        }
        /// \endcond

        /// Access the calling locality's value instance for this dist_object
        data_type const operator*() const
        {
            HPX_ASSERT(this->get_id());
            ensure_ptr();
            return **ptr;
        }

        /// Access the calling locality's value instance for this dist_object
        data_type operator*()
        {
            HPX_ASSERT(this->get_id());
            ensure_ptr();
            return **ptr;
        }

        /// Access the calling locality's value instance for this dist_object
        T const* operator->() const
        {
            HPX_ASSERT(this->get_id());
            ensure_ptr();
            return &**ptr;
        }

        /// Access the calling locality's value instance for this dist_object
        T* operator->()
        {
            HPX_ASSERT(this->get_id());
            ensure_ptr();
            return &**ptr;
        }

        /// Asynchronously returns a future of a copy of the instance of this
        /// dist_object associated with the given locality index. The locality
        /// index must be a valid locality ID with this dist_object.
        hpx::future<T> fetch(int idx)
        {
            HPX_ASSERT(this->get_id());
            hpx::id_type lookup = get_basename_helper(idx);
            typedef typename server::dist_object_part<T&>::fetch_ref_action
                action_type;
            return hpx::async<action_type>(lookup);
        }
        /// \cond NOINTERNAL
    private:
        mutable std::shared_ptr<server::dist_object_part<T&>> ptr;
        std::string base_;
        void ensure_ptr() const
        {
            if (!ptr)
            {
                ptr = hpx::get_ptr<server::dist_object_part<T&>>(
                    hpx::launch::sync, this->get_id());
            }
        }

    private:
        std::vector<hpx::id_type> basename_list;
        hpx::id_type get_basename_helper(int idx)
        {
            if (!basename_list[idx])
            {
                basename_list[idx] =
                    hpx::find_from_basename(base_ + std::to_string(idx), idx)
                        .get();
            }
            return basename_list[idx];
        }
        void basename_registration_helper(std::string base)
        {
            hpx::register_with_basename(
                base + std::to_string(hpx::get_locality_id()), this->get_id());
            basename_list.resize(hpx::find_all_localities().size());
        }
        /// \endcond
    };
}}}

#endif /*HPX_LCOS_DIST_OBJECT_HPP*/
