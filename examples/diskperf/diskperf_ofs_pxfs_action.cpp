
////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011-2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// use OrangeFS asyncio branch pxfs system call APIs and distribute with HPX actions

#include <hpx/hpx_init.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/serialization.hpp>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/shared_array.hpp>
#include <boost/thread/locks.hpp>

#include <fstream>
#include <cstdlib>
#include <ctime>
#include <sys/times.h>
#include <vector>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;
using hpx::util::high_resolution_timer;
using hpx::init;
using hpx::finalize;
using hpx::find_here;

//////////////////////////////////////////////////////////////////

#define MAXPROCS 1000


/* ------------------------  added pvfs header stuff --------------- */

#ifdef __cplusplus
extern "C" {
#endif

#include <pxfs.h>
// #include <orange.h>

#ifdef __cplusplus
} //extern "C" {
#endif

/* -------------------------  end pvfs header stuff --------------- */

typedef hpx::lcos::local::promise<int> int_promise_type;
typedef hpx::runtime rt_type;
typedef hpx::lcos::local::spinlock mutex_type;

struct RESULT {
    double real;
    double user;
    double sys;

    friend class hpx::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & real;
        ar & user;
        ar & sys;
    }
};

struct ofs_test_info_type
{
    uint64_t rfiles;
    uint64_t wfiles;
    ssize_t bufsiz;
    int count;
    std::string ofspath;
    std::string pvfs2tab_file;


    friend class hpx::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & rfiles;
        ar & wfiles;
        ar & bufsiz;
        ar & count;
        ar & ofspath;
        ar & pvfs2tab_file;
    }
};

struct promise_rt_ptr_type
{
    std::string thread_name_;
    int_promise_type* p_p_;
    rt_type* rt_p_;

    promise_rt_ptr_type() {}

    promise_rt_ptr_type(std::string name,
            int_promise_type* pp, rt_type* rr)
        : thread_name_(name), p_p_(pp), rt_p_(rr) {}

    void set(std::string name,
            int_promise_type* pp, rt_type* rr)
    {
        thread_name_ = name;
        p_p_ = pp;
        rt_p_ = rr;
    }
};

// this function will be executed by an HPX thread
void set_value(
    hpx::lcos::local::promise<int> *p,
    int result)
{
    // notify the waiting HPX thread and return a value
    p->set_value(result);
}

int set_promise_cb(void *cdat, int status)
{
    promise_rt_ptr_type *pr_p = (promise_rt_ptr_type *) cdat;

    pr_p->rt_p_->register_thread(pr_p->thread_name_.c_str());

//    pr_p->p_p_->set_value(status);
    // Create an HPX thread to guarantee that the promise::set_value
    // function can be invoked safely.
    hpx::threads::register_thread(hpx::util::bind(&set_value, pr_p->p_p_, status));

    pr_p->rt_p_->unregister_thread();

    return status;
}


///////////////////////////////////////////////////////////////////////////////
RESULT write_files_test(ofs_test_info_type ofs_test_info, int proc)
{
    // perform the I/O operations here
   char filename[1024];
   clock_t start;
   clock_t end;
   struct tms t1;
   struct tms t2;
   RESULT r;

   uint64_t wfiles = ofs_test_info.wfiles;
   ssize_t bufsiz = ofs_test_info.bufsiz;
   int count = ofs_test_info.count;
   std::string ofspath = ofs_test_info.ofspath;
   std::string pvfs2tab_file = ofs_test_info.pvfs2tab_file;

   // set the PVFS2TAB_FILE env
   std::string ofs_env_str = "PVFS2TAB_FILE=" + pvfs2tab_file;
   putenv((char*)ofs_env_str.c_str());

   boost::shared_array<int> fd_array(new int[wfiles]);
   boost::shared_array<char> buf(new char[bufsiz]);
   boost::shared_array<ssize_t> num_written_array(new ssize_t[count * wfiles]);
   std::vector<hpx::lcos::future<int> > futures;

   boost::shared_array<int_promise_type>
       int_promise_array(new int_promise_type[count * wfiles]);
   boost::shared_array<promise_rt_ptr_type>
       promise_rt_ptr_array(new promise_rt_ptr_type[count * wfiles]);

   srand((unsigned)time(0));
   for(ssize_t c = 0; c < bufsiz; c++)
       buf[c] = (unsigned int) rand() % 256;

   start = times(&t1);

   // buf should already been initialized
   for(boost::uint64_t i=0; i<wfiles; i++)
   {
       // using OrageFS syscalls
       int oflags;

       oflags = O_WRONLY|O_CREAT;
       sprintf(filename, "%s/loc_%d_file%d.%ld",
           ofspath.c_str(), hpx::get_locality_id(), proc, i);

       int_promise_type open_p;
       std::ostringstream sstream;
       sstream << "pxfs_open_loc" << hpx::get_locality_id() << "_ " << proc << "_" << i;
       promise_rt_ptr_type prp(sstream.str(), &open_p, hpx::get_runtime_ptr());

       pxfs_open(filename, oflags, &fd_array[i], set_promise_cb, &prp, 0777);
       open_p.get_future().get();// make sure pxfs_open finishes

// use synchronous call
//       fd_array[i] = pvfs_open(filename, oflags, 0777);

       if(fd_array[i] == -1)
       {
           hpx::cerr<<"Unable to open orangeFS file "<<filename<<hpx::endl;
           continue;
       }

       for (int j = 0; j < count; j++)
       {
           std::ostringstream tmp_sstream;
           tmp_sstream << "pxfs_pwrite_loc" << hpx::get_locality_id()
               << "_" << proc << "_" << i << "_" << j;
           promise_rt_ptr_array[i*count + j].set(
                   tmp_sstream.str(),
                   &int_promise_array[i*count + j],
                   hpx::get_runtime_ptr());

           pxfs_pwrite(fd_array[i], buf.get(), bufsiz, j*bufsiz,
           &num_written_array[i*count + j],
           set_promise_cb,
           &promise_rt_ptr_array[i*count + j]);

           futures.push_back(int_promise_array[i*count + j].get_future());
       }

   }

   hpx::lcos::wait_each(futures,
           hpx::util::unwrapped([&](int rt)
           {
           if (rt != 0)
           {
           hpx::cerr<<"loc " << hpx::get_locality_id() << " proc " << proc
           << ": error " << rt << " in writing one block of the file."
           <<hpx::endl;
           }

           }));

   end = times(&t2);

   for (uint64_t idx = 0; idx < count * wfiles; ++idx)
   {
       if(num_written_array[idx] != bufsiz)
       {
           hpx::cerr << "loc " << hpx::get_locality_id() << " proc " << proc
               << ": error! not writing all bytes of " << idx%count <<
               "th block of " << idx/count << "th file."<<hpx::endl;
       }
   }

   r.real = ((double) end - (double) start) / (double) sysconf(_SC_CLK_TCK);
   r.user = ((double) t2.tms_utime - (double) t1.tms_utime)
       / (double) sysconf(_SC_CLK_TCK);
   r.sys = ((double) t2.tms_stime - (double) t1.tms_stime)
       / (double) sysconf(_SC_CLK_TCK);

    for(uint64_t i=0; i<wfiles; i++)
    {
        pvfs_close(fd_array[i]);
    }

    return r;
}


HPX_PLAIN_ACTION(write_files_test, write_files_test_action);







///////////////////////////////////////////////////////////////////////////////
RESULT read_files_test(ofs_test_info_type ofs_test_info, int proc)
{
    // perform the I/O operations here
   char filename[1024];
   clock_t start;
   clock_t end;
   struct tms t1;
   struct tms t2;
   RESULT r;

   uint64_t rfiles = ofs_test_info.rfiles;
   ssize_t bufsiz = ofs_test_info.bufsiz;
   int count = ofs_test_info.count;
   std::string ofspath = ofs_test_info.ofspath;
   std::string pvfs2tab_file = ofs_test_info.pvfs2tab_file;

   // set the PVFS2TAB_FILE env
   std::string ofs_env_str = "PVFS2TAB_FILE=" + pvfs2tab_file;
   putenv((char*)ofs_env_str.c_str());

   boost::shared_array<int> fd_array(new int[rfiles]);
   boost::shared_array<char> buf_array(new char[bufsiz * count]);
   boost::shared_array<ssize_t> num_read_array(new ssize_t[count * rfiles]);
   std::vector<hpx::lcos::future<int> > futures;

   boost::shared_array<int_promise_type>
       int_promise_array(new int_promise_type[count * rfiles]);
   boost::shared_array<promise_rt_ptr_type>
       promise_rt_ptr_array(new promise_rt_ptr_type[count * rfiles]);

   start = times(&t1);

   // buf should already been allocated
   for(boost::uint64_t i=0; i<rfiles; i++)
   {
       // using OrageFS syscalls
       int oflags;

       oflags = O_RDONLY;
       sprintf(filename, "%s/loc_%d_file%d.%ld",
           ofspath.c_str(), hpx::get_locality_id(), proc, i);

       int_promise_type open_p;
       std::ostringstream sstream;
       sstream << "pxfs_open_loc" << hpx::get_locality_id() << "_ " << proc << "_" << i;
       promise_rt_ptr_type prp(sstream.str(), &open_p, hpx::get_runtime_ptr());

       pxfs_open(filename, oflags, &fd_array[i], set_promise_cb, &prp);
       open_p.get_future().get(); // make sure pxfs_open finishes

// use synchronous call
//       fd_array[i] = pvfs_open(filename, oflags);

       if(fd_array[i] == -1)
       {
           hpx::cerr<<"Unable to open orangeFS file "<<filename<<hpx::endl;
           continue;
       }

       for (int j = 0; j < count; j++)

       {
           std::ostringstream tmp_sstream;
           tmp_sstream << "pxfs_pwrite_loc" << hpx::get_locality_id()
               << "_" << proc << "_" << i << "_" << j;
           promise_rt_ptr_array[i*count + j].set(
                   tmp_sstream.str(),
                   &int_promise_array[i*count + j],
                   hpx::get_runtime_ptr());

           pxfs_pread(fd_array[i], &buf_array[j*bufsiz],
           bufsiz, j*bufsiz,
           &num_read_array[i*count + j],
           set_promise_cb,
           &promise_rt_ptr_array[i*count + j]);

           futures.push_back(int_promise_array[i*count + j].get_future());
       }
   }

   hpx::lcos::wait_each(futures,
           hpx::util::unwrapped([&](int rt)
           {
           if (rt != 0)
           {
           hpx::cerr<<"loc " << hpx::get_locality_id() << " proc " << proc
           << ": error " << rt << " in reading one block of the file."
           <<hpx::endl;
           }
           }));

   end = times(&t2);

   for (uint64_t idx = 0; idx < count * rfiles; ++idx)
   {
       if(num_read_array[idx] != bufsiz)
       {
           hpx::cerr << "loc " << hpx::get_locality_id() << " proc " << proc
               << ": error! not reading all bytes of " << idx%count
               <<"th block of " << idx/count << "th file."<<hpx::endl;
       }
   }


   r.real = ((double) end - (double) start) / (double) sysconf(_SC_CLK_TCK);
   r.user = ((double) t2.tms_utime - (double) t1.tms_utime)
       / (double) sysconf(_SC_CLK_TCK);
   r.sys = ((double) t2.tms_stime - (double) t1.tms_stime)
       / (double) sysconf(_SC_CLK_TCK);


   for(uint64_t i=0; i<rfiles; i++)
   {
       pvfs_close(fd_array[i]);
   }

   return r;
}


HPX_PLAIN_ACTION(read_files_test, read_files_test_action);


///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    // extract command line argument
    boost::uint64_t rfiles = vm["rfiles"].as<boost::uint64_t>();
    boost::uint64_t wfiles = vm["wfiles"].as<boost::uint64_t>();
    ssize_t      bufsiz = vm["bufsiz"].as<ssize_t>();
    int      count = vm["count"].as<int>();
    int      procs = vm["procs"].as<int>();
    bool is_remove = vm.count("remove");

    std::string ofspath;
    std::string pvfs2tab_file;
    ofs_test_info_type ofs_test_info;

    if(procs > MAXPROCS)
    {
        hpx::cerr << "too many thread numbers!" << hpx::endl;
        return hpx::finalize();
    }

    if(rfiles == 0 && wfiles == 0)
    {
        hpx::cerr << "need to specify either to read or write files" << hpx::endl;
        return hpx::finalize();
    }

    if(rfiles > 0 && wfiles > 0)
    {
        hpx::cerr << "Can't read and write in the same test" << hpx::endl;
        return hpx::finalize();
    }

    hpx::cout<<
        "Disk performance test with OrangeFS PXFS APIs and HPX actions."<<hpx::endl;

    if(vm.count("pvfs2tab"))
    {
        pvfs2tab_file = vm["pvfs2tab"].as<std::string>();
        hpx::cout<<"Using pvfs2tab file: "<<pvfs2tab_file<<hpx::endl;
    }
    else
    {
        hpx::cerr << "need to specify PVFS2TAB_FILE path!!" << hpx::endl;
        return hpx::finalize();
    }

    if(vm.count("ofsio"))
    {
        ofspath = vm["ofsio"].as<std::string>();
        hpx::cout<<"Using OrageFS path: "<<ofspath<<hpx::endl;
    }
    else
    {
        hpx::cerr << "need to specify OrangeFS path!!" << hpx::endl;
        return hpx::finalize();
    }

    // init ofs_test_info
    ofs_test_info.bufsiz = bufsiz;
    ofs_test_info.count = count;
    ofs_test_info.ofspath = ofspath;
    ofs_test_info.pvfs2tab_file = pvfs2tab_file;

    {
        // Find the localities connected to this application.
        std::vector<hpx::id_type> localities = hpx::find_all_localities();

        std::vector<hpx::lcos::future<RESULT> > futures;
        futures.reserve(procs * localities.size());

        std::vector<RESULT> result_vector;
        result_vector.reserve(procs * localities.size());
        double min_time = -1.0, max_time = -1.0;

        // Keep track of the time required to execute.
        high_resolution_timer t;

        if(rfiles > 0) // read tests
        {
            ofs_test_info.rfiles = rfiles;

            for (hpx::naming::id_type const& node : localities)
            {
                // Initiate an asynchronous IO operation wait for it to complete without
                // blocking any of the HPX thread-manager threads.
                for(int proc=0; proc < procs; proc++)
                {

                    futures.push_back(hpx::async<read_files_test_action>
                            (node, ofs_test_info, proc));
                }
            }
        }
        else // write test
        {
            ofs_test_info.wfiles = wfiles;

            for (hpx::naming::id_type const& node : localities)
            {
                for(int proc=0; proc < procs; proc++)
                {
                    futures.push_back(hpx::async<write_files_test_action>
                            (node, ofs_test_info, proc));
                }
            }
        }

        hpx::lcos::local::spinlock mtx;
        hpx::lcos::wait_each(futures,
                hpx::util::unwrapped([&](RESULT r) {
                    boost::lock_guard<hpx::lcos::local::spinlock> lk(mtx);
                    result_vector.push_back(r);
                }));

        // overall performance
        double tt = t.elapsed();

        if(rfiles > 0)
        {
            hpx::cout << (boost::format("%1% localities each has %2% threads,
                Reading %3% files") % localities.size() % procs % rfiles);
        }
        else
        {
            hpx::cout << (boost::format("%1% localities each has %2% threads,
                Writing %3% files") % localities.size() % procs % wfiles);
        }

        char const* fmt = " with count %1% x buffer size %2%M: \n";
        hpx::cout << (boost::format(fmt) % count % (bufsiz*1.0 /(1024*1024)) );

        for(uint64_t idx=0; idx < (localities.size() * procs); idx++)
        {
            int loc = idx / procs;
            int pr = idx % procs;

            hpx::cout << (boost::format("locality %1% thread %2% time:"
                        " Real %3% [s], User %4% [s], System %5% [s]\n") %
                    loc % pr % result_vector[idx].real %
                    result_vector[idx].user % result_vector[idx].sys);
            if ((min_time < 0) || (result_vector[idx].real < min_time) )
            {
                min_time = result_vector[idx].real;
            }
            if ((max_time < 0) || (result_vector[idx].real > max_time) )
            {
                max_time = result_vector[idx].real;
            }
        }

        hpx::cout<<"-------------------------------------------\n";
        hpx::cout<<(boost::format("Total elapsed time: %1% [s]\n") % tt);

        if(rfiles > 0)
        {
            hpx::cout<<(boost::format("Aggregate Reading Throughput: %1% [MB/s]\n") %
                    (localities.size() * procs * rfiles
                        * count * bufsiz / tt / (1024*1024)));
            hpx::cout<<(boost::format("\t Max Throughput per thread: %1% [MB/s]\n") %
                    (rfiles * count * bufsiz / min_time / (1024*1024)));
            hpx::cout<<(boost::format("\t Min Throughput per thread: %1% [MB/s]\n") %
                    (rfiles * count * bufsiz / max_time / (1024*1024)));
        }
        else
        {
            hpx::cout<<(boost::format("Aggregate Writing Throughput: %1% [MB/s]\n") %
                    (localities.size() * procs * wfiles
                        * count * bufsiz / tt / (1024*1024)));
            hpx::cout<<(boost::format("\t Max Throughput per thread: %1% [MB/s]\n") %
                    (wfiles * count * bufsiz / min_time / (1024*1024)));
            hpx::cout<<(boost::format("\t Min Throughput per thread: %1% [MB/s]\n") %
                    (wfiles * count * bufsiz / max_time / (1024*1024)));

        }

        hpx::cout<<hpx::endl;

        if(is_remove)
        {
            hpx::cout<<"removing test files ...\n";

            char filename[1024];
            uint64_t fileno = (rfiles > 0) ? rfiles : wfiles;

            for(uint32_t loc = 0; loc < localities.size(); ++loc)
            {
                for(int proc = 0; proc < procs; ++proc)
                {
                    for(uint64_t i = 0; i < fileno; ++i)
                    {
                        sprintf(filename, "%s/loc_%d_file%d.%ld",
                            ofspath.c_str(), loc, proc, i);
                        pvfs_unlink(filename);
                    }
                }
            }

            hpx::cout<<hpx::endl;

        }
    }



    return hpx::finalize();
}



///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "remove", "remove files after test.")
        ( "rfiles" , value<boost::uint64_t>()->default_value(0),
            "number of files to for reading")
        ( "wfiles" , value<boost::uint64_t>()->default_value(0),
            "number of files to for writing")
        ( "bufsiz" , value<ssize_t>()->default_value(10*1024),
            "buffer size (in bytes)")
        ( "count", value<int>()->default_value(1),
            "number of bufsiz in a file")
        ( "procs" , value<int>()->default_value(1),
            "number of threads used for processing")
        ( "ofsio" , value<std::string>(),
            "OrangeFS file path.")
        ( "pvfs2tab" , value<std::string>(),
            "PVFS2TAB file location.")
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

