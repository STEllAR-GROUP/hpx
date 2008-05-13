//  Copyright (c) 2007-2008 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
// 
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>

#include <boost/version.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>

#include <hpx/naming/locality.hpp>
#include <hpx/naming/resolver_client.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/parcelset/parcelport.hpp>
#include <hpx/parcelset/server/parcel_connection.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    parcelport::parcelport(naming::resolver_client& gas, 
            naming::locality here, std::size_t io_service_pool_size)
      : io_service_pool_(io_service_pool_size),
        acceptor_(io_service_pool_.get_io_service()),
        parcels_(This()),
        new_server_connection_(new server::connection(
              io_service_pool_.get_io_service(), parcels_)),
        resolver_(gas), here_(here), id_range_(here_, resolver_)
    {
        // retrieve the prefix to be used for this site
        resolver_.get_prefix(here, prefix_);    // throws on error
        
        // initialize network
        using boost::asio::ip::tcp;
        
        acceptor_.open(here.get_endpoint().protocol());
        acceptor_.set_option(tcp::acceptor::reuse_address(true));
        acceptor_.bind(here.get_endpoint());
        acceptor_.listen();
        acceptor_.async_accept(new_server_connection_->socket(),
            boost::bind(&parcelport::handle_accept, this,
                boost::asio::placeholders::error));
    }

    parcelport::parcelport(naming::resolver_client& gas, 
            std::string address, unsigned short port, 
            std::size_t io_service_pool_size)
      : io_service_pool_(io_service_pool_size),
        acceptor_(io_service_pool_.get_io_service()),
        parcels_(This()),
        new_server_connection_(new server::connection(
              io_service_pool_.get_io_service(), parcels_)),
        resolver_(gas), here_(address, port), 
        id_range_(here_, resolver_)
    {
        // retrieve the prefix to be used for this site
        resolver_.get_prefix(here_, prefix_);    // throws on error
        
        // initialize network
        using boost::asio::ip::tcp;
        
        // Open the acceptor with the option to reuse the address (i.e. SO_REUSEADDR).
        tcp::resolver resolver(acceptor_.io_service());
        tcp::resolver::query query(address, 
            boost::lexical_cast<std::string>(port));
        tcp::endpoint endpoint = *resolver.resolve(query);

        acceptor_.open(endpoint.protocol());
        acceptor_.set_option(tcp::acceptor::reuse_address(true));
        acceptor_.bind(endpoint);
        acceptor_.listen();
        acceptor_.async_accept(new_server_connection_->socket(),
            boost::bind(&parcelport::handle_accept, this,
                boost::asio::placeholders::error));
    }

    void parcelport::run(bool blocking)
    {
        io_service_pool_.run(blocking);
    }

    void parcelport::stop()
    {
        io_service_pool_.stop();
        io_service_pool_.join();
    }
    
    /// accepted new incoming connection
    void parcelport::handle_accept(boost::system::error_code const& e)
    {
        if (!e) {
        // handle this incoming parcel
            new_server_connection_->async_read(
                boost::bind(&parcelport::handle_read_completion, this,
                boost::asio::placeholders::error));

        // create new connection waiting for next incoming parcel
            new_server_connection_.reset(new server::connection(
                io_service_pool_.get_io_service(), parcels_));
            acceptor_.async_accept(new_server_connection_->socket(),
                boost::bind(&parcelport::handle_accept, this,
                    boost::asio::placeholders::error));
        }
    }

    /// Handle completion of a read operation.
    void parcelport::handle_read_completion(boost::system::error_code const& e)
    {
        if (e && e != boost::asio::error::operation_aborted)
        {
            // FIXME: add error handling
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // A parcel is submitted for transport at the source locality site to 
    // the parcel set of the locality with the put-parcel command
    // This function is synchronous.
    
    struct wait_for_put_parcel
    {
        typedef boost::mutex mutex_type;
        typedef boost::condition condition_type;

        wait_for_put_parcel(mutex_type& mtx, condition_type& cond,
              boost::system::error_code& saved_error, 
              bool& waiting, bool& finished)
          : mtx_(mtx), cond_(cond), saved_error_(saved_error),
            waiting_(waiting), finished_(finished)
        {}
        
        void operator()(boost::system::error_code const& e, std::size_t size)
        {
            mutex_type::scoped_lock l(mtx_);
            if (e) 
                saved_error_ = e;
                
            if (waiting_)
                cond_.notify_one();
            finished_ = true;
        }

        bool wait()
        {
            mutex_type::scoped_lock l(mtx_);
            
            if (finished_) 
                return true;
                
            boost::xtime xt;
            boost::xtime_get(&xt, boost::TIME_UTC);
            xt.sec += 5;        // wait for max. 5sec

            waiting_ = true;
            return cond_.timed_wait(l, xt);
        }
        
        mutex_type& mtx_;
        condition_type& cond_;
        boost::system::error_code& saved_error_;
        bool& waiting_;
        bool& finished_;
    };
    
    parcel_id parcelport::sync_put_parcel(parcel& p)
    {
        wait_for_put_parcel::mutex_type mtx;
        wait_for_put_parcel::condition_type cond;
        boost::system::error_code saved_error;
        bool waiting = false, finished = false;
        
        wait_for_put_parcel wfp (mtx, cond, saved_error, waiting, finished);
        parcel_id id = put_parcel(p, wfp);  // schedule parcel send
        if (!wfp.wait())                    // wait for the parcel being sent
            throw exception(network_error, "timeout");
            
        if (saved_error) 
            throw exception(network_error, saved_error.message());
        return id;
    }
    
///////////////////////////////////////////////////////////////////////////////
}}
