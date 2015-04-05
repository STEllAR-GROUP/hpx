//  Copyright (c) 2013-2014 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SIMPLE_CENTRAL_TUPLESPACE_MAR_31_2013_0555PM)
#define HPX_SIMPLE_CENTRAL_TUPLESPACE_MAR_31_2013_0555PM

#include <hpx/include/components.hpp>

#include "stubs/simple_central_tuplespace.hpp"

namespace examples
{
    ///////////////////////////////////////////////////////////////////////////
    /// Client for the \a server::simple_central_tuplespace component.
    //[simple_central_tuplespace_client_inherit
    class simple_central_tuplespace
      : public hpx::components::client_base<
            simple_central_tuplespace, stubs::simple_central_tuplespace
        >
    //]
    {
        //[simple_central_tuplespace_base_type
        typedef hpx::components::client_base<
            simple_central_tuplespace, stubs::simple_central_tuplespace
        > base_type;
        //]

        typedef base_type::tuple_type tuple_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        simple_central_tuplespace()
        {}

        /// Create a client side representation for the existing
        /// \a server::simple_central_tuplespace instance with the given GID.
        simple_central_tuplespace(hpx::shared_future<hpx::naming::id_type> const& gid)
          : base_type(gid)
        {}

        simple_central_tuplespace(hpx::naming::id_type const& gid)
          : base_type(gid)
        {}

        bool create(std::string const& symbol_name, hpx::id_type const& locality)
        {
            if(!symbol_name_.empty())
            {
                hpx::cerr<<"simple_central_tuplespace::create() : ERROR! current instance not empty!\n";
                return false;
            }
            if(symbol_name_ == symbol_name) // itself
            {
                hpx::cerr<<"simple_central_tuplespace::create() : ERROR! current instance already attached to "<< symbol_name <<"\n";
                return false;
            }

            // request gid;
            *this = simple_central_tuplespace(hpx::components::new_<examples::server::simple_central_tuplespace>(locality));
            bool rc = hpx::agas::register_name(symbol_name, this->get_gid()).get();

            if(rc)
            {
                symbol_name_ = symbol_name;
            }

            return rc;
        }

        bool connect(std::string const& symbol_name)
        {
            if(symbol_name_ == symbol_name)
            {
                hpx::cerr<<"simple_central_tuplespace::connect() : ERROR! current instance already attached to "<< symbol_name <<"\n";
                return false;
            }

            *this = hpx::agas::resolve_name(symbol_name).get();

            symbol_name_ = symbol_name;

            return true;
        }



        ///////////////////////////////////////////////////////////////////////
        /// put \p tuple into tuplespace.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        //[simple_central_tuplespace_client_write_async
        hpx::lcos::future<int> write_async(const tuple_type& tuple)
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::write_async(this->get_gid(), tuple);
        }
        //]

        /// put \p tuple into tuplespace.
        ///
        /// \note This function is fully synchronous.
        int write_sync(const tuple_type& tuple)
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::write_sync(this->get_gid(), tuple);
        }

        ///////////////////////////////////////////////////////////////////////
        /// read matching tuple from tuplespace within \p timeout.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        hpx::lcos::future<tuple_type> read_async(const tuple_type& tp, long const timeout)
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::read_async(this->get_gid(), tp, timeout);
        }

        /// read matching tuple from tuplespace within \p timeout.
        ///
        /// \note This function is fully synchronous.
        //[simple_central_tuplespace_client_read_sync
        tuple_type read_sync(const tuple_type& tp, long const timeout)
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::read_sync(this->get_gid(), tp, timeout);
        }
        //]

        ///////////////////////////////////////////////////////////////////////
        /// take matching tuple from tuplespace within \p timeout.
        ///
        /// \returns This function returns an \a hpx::lcos::future. When the
        ///          value of this computation is needed, the get() method of
        ///          the future should be called. If the value is available,
        ///          get() will return immediately; otherwise, it will block
        ///          until the value is ready.
        //[simple_central_tuplespace_client_take_async
        hpx::lcos::future<tuple_type> take_async(const tuple_type& tp, long const timeout)
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::take_async(this->get_gid(), tp, timeout);
        }
        //]

        /// take matching tuple from tuplespace within \p timeout.
        ///
        /// \note This function is fully synchronous.
        tuple_type take_sync(const tuple_type& tp, long const timeout)
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::take_sync(this->get_gid(), tp, timeout);
        }

        ///////////////////////////////////////////////////////////////////////
        /// store the tuplespace into disk.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        //[simple_central_tuplespace_client_store_async
        hpx::lcos::future<int> store_async(const std::string& file_name)
        {
            HPX_ASSERT(this->get_gid());
            file_name_ = file_name;
            return this->base_type::store_async(this->get_gid(), file_name);
        }
        //]

        /// store the tuplespace into disk.
        ///
        /// \note This function is fully synchronous.
        int store_sync(const std::string& file_name)
        {
            HPX_ASSERT(this->get_gid());
            file_name_ = file_name;
            return this->base_type::store_sync(this->get_gid(), file_name);
        }

        ///////////////////////////////////////////////////////////////////////
        /// load the tuplespace from disk.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        //[simple_central_tuplespace_client_load_async
        hpx::lcos::future<int> load_async(const std::string& file_name)
        {
            HPX_ASSERT(this->get_gid());
            file_name_ = file_name;
            return this->base_type::load_async(this->get_gid(), file_name);
        }
        //]

        /// load the tuplespace into disk.
        ///
        /// \note This function is fully synchronous.
        int load_sync(const std::string& file_name)
        {
            HPX_ASSERT(this->get_gid());
            file_name_ = file_name;
            return this->base_type::load_sync(this->get_gid(), file_name);
        }

        /// load the tuplespace into disk.
        ///
        /// \note This function is fully synchronous.
        std::string print()
        {
            HPX_ASSERT(this->get_gid());

            std::stringstream ss;
            ss << "TupleSpace Symbol Name: " << symbol_name_ << "\n";
            ss << this->base_type::print_sync(this->get_gid());

            return ss.str();
        }

        ///////////////////////////////////////////////////////////////////////
        /// clear the tuplespace contents.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        //[simple_central_tuplespace_client_clear_async
        hpx::lcos::future<void> clear_async()
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::clear_async(this->get_gid());
        }
        //]

        /// clear the tuplespace contents.
        ///
        /// \note This function is fully synchronous.
        void clear_sync()
        {
            HPX_ASSERT(this->get_gid());
            return this->base_type::clear_sync(this->get_gid());
        }

    private:
        std::string symbol_name_;
        std::string file_name_;
    };
} // examples

#endif

