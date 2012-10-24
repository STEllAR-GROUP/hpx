//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_COMPONENTS_MEMORY_BLOCK_OCT_22_2008_0416PM)
#define HPX_COMPONENTS_MEMORY_BLOCK_OCT_22_2008_0416PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/components/stubs/memory_block.hpp>
#include <hpx/include/client.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/is_const.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    template <typename T>
    class access_memory_block;

    ///////////////////////////////////////////////////////////////////////////
    /// The \a runtime_support class is the client side representation of a
    /// \a server#memory_block component
    class memory_block : public client_base<memory_block, stubs::memory_block>
    {
    private:
        typedef client_base<memory_block, stubs::memory_block> base_type;

    public:
        memory_block() {}

        /// Create a client side representation for the existing
        /// \a server#memory_block instance with the given global id \a gid.
        memory_block(naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        /// Create a new instance of a memory_block component on the locality as
        /// given by the parameter \a targetgid
//         template <typename T>
//         memory_block& create(naming::id_type const& targetgid, std::size_t count,
//             hpx::actions::manage_object_action<T> const& act)
//         {
//             this->base_type::free();
//             gid_ = stub_type::create(targetgid, count, act);
//             return *this;
//         }

        template <typename T, typename Config>
        memory_block& create(naming::id_type const& targetgid, std::size_t count,
            hpx::actions::manage_object_action<T, Config> const& act)
        {
            this->base_type::free();
            gid_ = stub_type::create(targetgid, count, act);
            return *this;
        }

        /// Create a new instance of a memory_block component on the locality as
        /// given by the parameter \a targetgid.
        /// Allocates count * sizeof(T1) bytes of memory
        template <typename T0, typename T1, typename Config>
        memory_block& create(naming::id_type const& targetgid, std::size_t count,
            hpx::actions::manage_object_action<T1, Config> const& act)
        {
            return create(targetgid, sizeof(T0) * count, act);
        }

        /// Create a new instance of a memory_block component on the locality as
        /// given by the parameter \a targetgid.
        /// Allocates count * sizeof(T) bytes of memory and automatically creates
        /// an instance of hpx::actions::manage_object_action<T>
        template <typename T>
        memory_block& create(naming::id_type const& targetgid, std::size_t count)
        {
            hpx::actions::manage_object_action<T> const act;
            return create(targetgid, sizeof(T) * count, act);
        }

        template <typename T0, typename T1>
        memory_block& create(naming::id_type const& targetgid, std::size_t count)
        {
            hpx::actions::manage_object_action<T1> const act;
            return create(targetgid, sizeof(T0) * count, act);
        }

        template <typename T0, typename T1, typename Config>
        memory_block& create(naming::id_type const& targetgid, std::size_t count)
        {
            hpx::actions::manage_object_action<T1, Config> const act;
            return create(targetgid, sizeof(T0) * count, act);
        }


        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Get the \a memory_block_data maintained by this memory_block
        memory_block_data get()
        {
            return this->base_type::get(get_gid());
        }

        /// Asynchronously get the \a memory_block_data maintained by this
        /// memory_block
        lcos::future<memory_block_data> get_async()
        {
            return this->base_type::get_async(get_gid());
        }

        /// Get the \a memory_block_data maintained by this memory_block, use
        /// given data for serialization configuration (will be passed to the
        /// save() function exposed by the datatype instance wrapped in the
        /// return value of this get())
        memory_block_data get(memory_block_data const& config)
        {
            return this->base_type::get(get_gid(), config);
        }

        /// Asynchronously get the \a memory_block_data maintained by this
        /// memory_block. Use given data for serialization configuration (will
        /// be passed to the save() function exposed by the datatype instance
        /// wrapped in the return value of this get())
        lcos::future<memory_block_data> get_async(
            memory_block_data const& config)
        {
            return this->base_type::get_async(get_gid(), config);
        }

        ///////////////////////////////////////////////////////////////////////
        /// Clone the \a memory_block_data maintained by this memory_block
        naming::id_type clone()
        {
            return this->base_type::clone(get_gid());
        }

        /// Asynchronously clone the \a memory_block_data maintained by this
        /// memory_block
        lcos::future<naming::id_type, naming::gid_type> clone_async()
        {
            return this->base_type::clone_async(get_gid());
        }

        ///////////////////////////////////////////////////////////////////////
        /// Write the given \a memory_block_data back to it's original source
        template <typename T>
        void checkin(components::access_memory_block<T> const& data)
        {
            this->base_type::checkin(get_gid(), data);
        }

        /// Asynchronously clone the \a memory_block_data maintained by this
        /// memory_block
        template <typename T>
        lcos::future<void> checkin_async(
            components::access_memory_block<T> const& data)
        {
            return this->base_type::checkin_async(get_gid(), data);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class access_memory_block_proxy
    {
    private:
        typedef typename boost::remove_const<T>::type target_type;
        typedef typename
            boost::mpl::if_<
                boost::is_const<T>, memory_block_data const&, memory_block_data&
            >::type
        wrapped_type;

    public:
        explicit access_memory_block_proxy(wrapped_type block)
          : block_(block)
        {}

        access_memory_block_proxy& operator=(target_type const& rhs)
        {
            block_.template set<target_type>(rhs);
            return *this;
        }

        operator target_type const&() const
        {
            return block_.template get<target_type>();
        }

    private:
        wrapped_type block_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class access_memory_block
    {
    private:
        typedef typename boost::remove_const<T>::type target_type;

    public:
        access_memory_block()
        {}
        access_memory_block(memory_block_data const& mb)
          : mb_(mb)
        {}

        access_memory_block& operator=(memory_block_data const& mb)
        {
            mb_ = mb;
            return *this;
        }

        ///////////////////////////////////////////////////////////////////////
        access_memory_block_proxy<T> operator*()
        {
            return access_memory_block_proxy<T>(mb_);
        }
        access_memory_block_proxy<T const> operator*() const
        {
            return access_memory_block_proxy<T>(mb_);
        }

        ///////////////////////////////////////////////////////////////////////
        target_type* operator->()
        {
            BOOST_ASSERT(sizeof(target_type) <= mb_.get_size());
            return reinterpret_cast<target_type*>(mb_.get_ptr());
        }
        target_type const* operator->() const
        {
            return reinterpret_cast<target_type const*>(mb_.get_ptr());
        }

        ///////////////////////////////////////////////////////////////////////
        target_type* get_ptr()
        {
            BOOST_ASSERT(sizeof(target_type) <= mb_.get_size());
            return reinterpret_cast<target_type*>(mb_.get_ptr());
        }

        target_type const* get_ptr() const
        {
            BOOST_ASSERT(sizeof(target_type) <= mb_.get_size());
            return reinterpret_cast<target_type const*>(mb_.get_ptr());
        }

        ///////////////////////////////////////////////////////////////////////
        target_type& get()
        {
            BOOST_ASSERT(sizeof(target_type) <= mb_.get_size());
            return *reinterpret_cast<target_type*>(mb_.get_ptr());
        }

        target_type const& get() const
        {
            BOOST_ASSERT(sizeof(target_type) <= mb_.get_size());
            return *reinterpret_cast<target_type const*>(mb_.get_ptr());
        }

        memory_block_data const& get_memory_block() const { return mb_; }

    private:
        memory_block_data mb_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // helper functions to get several memory pointers asynchronously

    template <typename T, typename AllocA, typename AllocB>
    inline void
    get_memory_block_async(std::vector<access_memory_block<T>, AllocA>& results,
        std::vector<naming::id_type, AllocB> const& gids)
    {
        typedef std::vector<lcos::future<memory_block_data>,
            typename AllocA::template rebind<
                lcos::future<memory_block_data>
            >::other
        > lazy_results_type;
        lazy_results_type lazy_results;

        // first invoke all remote operations
        typedef typename std::vector<naming::id_type, AllocB>::const_iterator
            const_iterator_type;

        const_iterator_type end = gids.end();
        for (const_iterator_type it = gids.begin(); it != end; ++it)
            lazy_results.push_back(stubs::memory_block::get_async(*it));

        // then wait for all results to get back to us
        typedef typename lazy_results_type::iterator iterator_type;
        iterator_type lend = lazy_results.end();
        for (iterator_type lit = lazy_results.begin(); lit != lend; ++lit)
            results.push_back((*lit).get());
    }

    template <typename T, typename AllocA, typename AllocB>
    inline access_memory_block<T>
    get_memory_block_async(std::vector<access_memory_block<T>, AllocA>& results,
        std::vector<naming::id_type, AllocB> const& gids,
        naming::id_type const& result)
    {
        typedef std::vector<lcos::future<memory_block_data>,
            typename AllocA::template rebind<
                lcos::future<memory_block_data>
            >::other
        > lazy_results_type;
        lazy_results_type lazy_results;

        // first invoke all remote operations
        typedef typename std::vector<naming::id_type, AllocB>::const_iterator
            const_iterator_type;

        const_iterator_type end = gids.end();
        for (const_iterator_type it = gids.begin(); it != end; ++it)
            lazy_results.push_back(stubs::memory_block::get_async(*it));

        //  invoke the remote operation for the result gid as well
        lcos::future<memory_block_data> lazy_result =
            stubs::memory_block::get_async(result);

        // then wait for all results to get back to us
        typedef typename lazy_results_type::iterator iterator_type;
        iterator_type lend = lazy_results.end();
        for (iterator_type lit = lazy_results.begin(); lit != lend; ++lit)
            results.push_back((*lit).get());

        // now return the resolved result
        return lazy_result.get();
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/components/preprocessed/memory_block.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/memory_block_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_WAIT_ARGUMENT_LIMIT,                                          \
    "hpx/runtime/components/memory_block.hpp"))                               \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()
#define HPX_ACCESS_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n)                  \
        access_memory_block<T>                                                \
    /**/
#define HPX_GET_ASYNC_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n)               \
        naming::id_type const& BOOST_PP_CAT(g, n)                            \
    /**/
#define HPX_WAIT_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n)                    \
        stubs::memory_block::get_async(BOOST_PP_CAT(g, n))                    \
    /**/

namespace hpx { namespace components
{
    template <typename T>
    inline HPX_STD_TUPLE<BOOST_PP_REPEAT(N, HPX_ACCESS_ARGUMENT, _)>
    get_memory_block_async(BOOST_PP_REPEAT(N, HPX_GET_ASYNC_ARGUMENT, _))
    {
        return lcos::wait(BOOST_PP_REPEAT(N, HPX_WAIT_ARGUMENT, _));
    }
}}

#undef HPX_WAIT_ARGUMENT
#undef HPX_GET_ASYNC_ARGUMENT
#undef HPX_ACCESS_ARGUMENT
#undef N

#endif
