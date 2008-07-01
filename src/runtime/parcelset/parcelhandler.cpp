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

#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/threadmanager/threadmanager.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
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
    
    parcel_id parcelhandler::sync_put_parcel(parcel& p)
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

    void parcelhandler::parcel_sink(parcelport& pp, 
        boost::shared_ptr<std::vector<char> > const& parcel_data)
    {
        if (NULL == tm_) {
            // this is supported for debugging purposes mainly, it results in
            // the direct execution of the parcel decoding
            decode_parcel(parcel_data);
        }
        else {
            // create a new px_thread which decodes and handles the parcel
            tm_->register_work(
                boost::bind(&parcelhandler::decode, this, _1, parcel_data));
        }
    }

    threadmanager::thread_state parcelhandler::decode(
        threadmanager::px_thread_self&, 
        boost::shared_ptr<std::vector<char> > const& parcel_data)
    {
        decode_parcel(parcel_data);
        return threadmanager::terminated;
    }

    void parcelhandler::decode_parcel(
        boost::shared_ptr<std::vector<char> > const& parcel_data)
    {
        parcel p;
        {
            // create a special io stream on top of in_buffer_
            typedef util::container_device<std::vector<char> > io_device_type;
            boost::iostreams::stream<io_device_type> io (*parcel_data.get());

            // De-serialize the parcel data
            util::portable_binary_iarchive archive(io);
            archive >> p;
        }

        // add parcel to incoming parcel queue
        parcels_.add_parcel(p);
    }

///////////////////////////////////////////////////////////////////////////////
}}
