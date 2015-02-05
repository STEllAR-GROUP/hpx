////////////////////////////////////////////////////////////////////////////////
//  Copyright (c)      2014 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// disk performance benchmark using hpx and async APIs of orangefs_file and
// pxfs_file classes

#include <hpx/components/io/orangefs_file.hpp>
#include <hpx/components/io/pxfs_file.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <fstream>
#include <cstdlib>
#include <ctime>
#include <sys/times.h>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/shared_array.hpp>

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

struct RESULT {
    double	real;
    double	user;
    double	sys;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & real;
        ar & user;
        ar & sys;
    }
};

struct test_info_type
{
    uint64_t rfiles;
    uint64_t wfiles;
    ssize_t bufsiz;
    int count;
    std::string path;
    bool is_remove;
    ssize_t procs;

    test_info_type() {}

    test_info_type(uint64_t rf, uint64_t wf, ssize_t size,
            int cc, std::string p, bool remove, ssize_t ps) :
        rfiles(rf), wfiles(wf), bufsiz(size),
        count(cc), path(p), is_remove(remove), procs(ps) {}

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & rfiles;
        ar & wfiles;
        ar & bufsiz;
        ar & count;
        ar & path;
        ar & is_remove;
        ar & procs;
    }
};

///////////////////////////////////////////////////////////////////////////////
RESULT orangefs_file_test(test_info_type const& test_info, int const proc)
{
    // perform the I/O operations here
   char filename[1024];
   clock_t	start;
   clock_t	end;
   struct tms	t1;
   struct tms	t2;
   RESULT r;
   std::vector<hpx::io::orangefs_file> of_vector;

   if (test_info.wfiles > 0)
   { // writing test
       boost::shared_array<char> buf(new char[test_info.bufsiz]);
       std::vector<hpx::lcos::future<ssize_t> > futures;
       of_vector.reserve(test_info.wfiles);

       srand((unsigned)time(0));

       start = times(&t1);

       for(boost::uint64_t i=0; i<test_info.wfiles; i++)
       {
           sprintf(filename, "%s/loc_%d_file%d.%ld", test_info.path.c_str(),
                   hpx::get_locality_id(), proc, i);

           of_vector.push_back(hpx::io::orangefs_file::create(hpx::find_here()));

           of_vector[i].open_sync(std::string(filename), O_WRONLY|O_CREAT);

           if (!of_vector[i].is_open_sync())
           {
               hpx::cerr << "Unable to open orangefs file " <<
                   filename << hpx::endl;
               continue;
           }

           for (int j = 0; j < test_info.count; ++j)
           {

               for(ssize_t c = 0; c < test_info.bufsiz; c++)
                   buf[c] = (unsigned int) rand() % 256;

               futures.push_back(of_vector[i].pwrite(
                           std::vector<char>(buf.get(),
                               buf.get() + test_info.bufsiz),
                           j * test_info.bufsiz));
           }
       }

       hpx::lcos::wait_each(
               hpx::util::unwrapped([&](ssize_t rt)
                   {
                   if (rt != test_info.bufsiz)
                   {
                   hpx::cerr << "loc " << hpx::get_locality_id() << " proc " << proc
                   << ": error! not writing all bytes of one block ."<<hpx::endl;
                   }
                   }),
               futures);
       end = times(&t2);
   }
   else
   { // reading test
       std::vector<hpx::lcos::future<std::vector<char> > > futures;
       of_vector.reserve(test_info.rfiles);

       srand((unsigned)time(0));

       start = times(&t1);

       for(boost::uint64_t i=0; i<test_info.rfiles; i++)
       {
           sprintf(filename, "%s/loc_%d_file%d.%ld", test_info.path.c_str(),
                   hpx::get_locality_id(), proc, i);
           of_vector.push_back(hpx::io::orangefs_file::create(hpx::find_here()));
           of_vector[i].open_sync(std::string(filename), O_RDONLY);

           if (!of_vector[i].is_open_sync())
           {
               hpx::cerr << "Unable to open orangefs file " <<
                   filename << hpx::endl;
               continue;
           }

           for (int j = 0; j < test_info.count; ++j)
           {
               futures.push_back(of_vector[i].pread(
                           test_info.bufsiz, j * test_info.bufsiz));
           }
       }

       hpx::lcos::wait_each(
               hpx::util::unwrapped([&](std::vector<char> buf)
                   {
                   if (static_cast<ssize_t>(buf.size()) != test_info.bufsiz)
                   {
                   hpx::cerr << "loc " << hpx::get_locality_id() << " proc "
                   << proc << ": error! not reading all bytes of one block ."
                   << hpx::endl;
                   }
                   }),
               futures);

       end = times(&t2);
   }

   r.real = ((double) end - (double) start) / (double) sysconf(_SC_CLK_TCK);
   r.user = ((double) t2.tms_utime - (double) t1.tms_utime) / (double) sysconf(_SC_CLK_TCK);
   r.sys = ((double) t2.tms_stime - (double) t1.tms_stime) / (double) sysconf(_SC_CLK_TCK);

   // destructor of orangefs_file will close the files
   return r;
}

HPX_PLAIN_ACTION(orangefs_file_test, orangefs_file_test_action);


RESULT pxfs_file_test(test_info_type const& test_info, int const proc)
{
    // perform the I/O operations here
   char filename[1024];
   clock_t	start;
   clock_t	end;
   struct tms	t1;
   struct tms	t2;
   RESULT r;
   std::vector<hpx::io::pxfs_file> pf_vector;

   if (test_info.wfiles > 0)
   { // writing test
       boost::shared_array<char> buf(new char[test_info.bufsiz]);
       std::vector<hpx::lcos::future<ssize_t> > futures;
       pf_vector.reserve(test_info.wfiles);

       srand((unsigned)time(0));

       start = times(&t1);

       for(boost::uint64_t i=0; i<test_info.wfiles; i++)
       {
           sprintf(filename, "%s/loc_%d_file%d.%ld", test_info.path.c_str(),
                   hpx::get_locality_id(), proc, i);

           pf_vector.push_back(hpx::io::pxfs_file());

           pf_vector[i].open_sync(std::string(filename), O_WRONLY|O_CREAT);

           if (!pf_vector[i].is_open_sync())
           {
               hpx::cerr << "Unable to open pxfs file " <<
                   filename << hpx::endl;
               continue;
           }

           for (int j = 0; j < test_info.count; ++j)
           {

               for(ssize_t c = 0; c < test_info.bufsiz; c++)
                   buf[c] = (unsigned int) rand() % 256;

               futures.push_back(pf_vector[i].pwrite(
                           std::vector<char>(buf.get(),
                               buf.get() + test_info.bufsiz),
                           j * test_info.bufsiz));
           }
       }

       hpx::lcos::wait_each(
               hpx::util::unwrapped([&](ssize_t rt)
                   {
                   if (rt != test_info.bufsiz)
                   {
                   hpx::cerr << "loc " << hpx::get_locality_id() << " proc " << proc
                   << ": error! not writing all bytes of one block ."<<hpx::endl;
                   }
                   }),
               futures);
       end = times(&t2);
   }
   else
   { // reading test
       std::vector<hpx::lcos::future<std::vector<char> > > futures;
       pf_vector.reserve(test_info.rfiles);

       srand((unsigned)time(0));

       start = times(&t1);

       for(boost::uint64_t i=0; i<test_info.rfiles; i++)
       {
           sprintf(filename, "%s/loc_%d_file%d.%ld", test_info.path.c_str(),
                   hpx::get_locality_id(), proc, i);
           pf_vector.push_back(hpx::io::pxfs_file());
           pf_vector[i].open_sync(std::string(filename), O_RDONLY);

           if (!pf_vector[i].is_open_sync())
           {
               hpx::cerr << "Unable to open pxfs file " <<
                   filename << hpx::endl;
               continue;
           }

           for (int j = 0; j < test_info.count; ++j)
           {
               futures.push_back(pf_vector[i].pread(
                           test_info.bufsiz, j * test_info.bufsiz));
           }
       }

       hpx::lcos::wait_each(
               hpx::util::unwrapped([&](std::vector<char> buf)
                   {
                   if (static_cast<ssize_t>(buf.size()) != test_info.bufsiz)
                   {
                   hpx::cerr << "loc " << hpx::get_locality_id() << " proc "
                   << proc << ": error! not reading all bytes of one block ."
                   << hpx::endl;
                   }
                   }),
               futures);

       end = times(&t2);
   }

   r.real = ((double) end - (double) start) / (double) sysconf(_SC_CLK_TCK);
   r.user = ((double) t2.tms_utime - (double) t1.tms_utime) / (double) sysconf(_SC_CLK_TCK);
   r.sys = ((double) t2.tms_stime - (double) t1.tms_stime) / (double) sysconf(_SC_CLK_TCK);

   // destructor of orangefs_file will close the files
   return r;
}

HPX_PLAIN_ACTION(pxfs_file_test, pxfs_file_test_action);


///////////////////////////////////////////////////////////////////////////////
void run_orangefs_file_test(test_info_type const& test_info)
{
    // Find the localities connected to this application.
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    std::vector<hpx::lcos::future<RESULT> > futures;
    futures.reserve(test_info.procs * localities.size());

    std::vector<RESULT> result_vector;
    result_vector.reserve(test_info.procs * localities.size());
    double min_time = -1.0, max_time = -1.0;

    // Keep track of the time required to execute.
    high_resolution_timer t;

    BOOST_FOREACH(hpx::naming::id_type const& node, localities)
    {
        // Initiate an asynchronous IO operation wait for it to complete without
        // blocking any of the HPX thread-manager threads.
        for(ssize_t proc=0; proc < test_info.procs; proc++)
        {
            futures.push_back(hpx::async<orangefs_file_test_action>
                    (node, test_info, proc));
        }
    }

    hpx::lcos::local::spinlock mtx;
    hpx::lcos::wait_each(
            hpx::util::unwrapped([&](RESULT r) {
                hpx::lcos::local::spinlock::scoped_lock lk(mtx);
                result_vector.push_back(r);
                }),
            futures);

    // overall performance
    double tt = t.elapsed();

    hpx::cout << "orangefs_file performance results:" << hpx::endl;

    if(test_info.rfiles > 0)
    {
        hpx::cout << (boost::format("%1% localities each has %2% threads,"
                    " Reading %3% files") % localities.size() %
                test_info.procs % test_info.rfiles);
    }
    else
    {
        hpx::cout << (boost::format("%1% localities each has %2% threads,"
                    " Writing %3% files") % localities.size() %
                    test_info.procs % test_info.wfiles);
    }

    char const* fmt = " with count %1% x buffer size %2%M: \n";
    hpx::cout << (boost::format(fmt) % test_info.count %
            (test_info.bufsiz * 1.0 / (1024 * 1024)));

    for(uint64_t idx=0; idx < (localities.size() * test_info.procs); idx++)
    {
        int loc = idx / test_info.procs;
        int pr = idx % test_info.procs;

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

    if(test_info.rfiles > 0)
    {
        hpx::cout << (boost::format(
                    "Aggregate Reading Throughput: %1% [MB/s]\n") %
                (localities.size() * test_info.procs * test_info.rfiles *
                 test_info.count * test_info.bufsiz / tt / (1024*1024)));
        hpx::cout << (boost::format(
                    "\t Max Throughput per thread: %1% [MB/s]\n") %
                (test_info.rfiles * test_info.count * test_info.bufsiz /
                 min_time / (1024*1024)));
        hpx::cout << (boost::format(
                    "\t Min Throughput per thread: %1% [MB/s]\n") %
                (test_info.rfiles * test_info.count * test_info.bufsiz /
                 max_time / (1024*1024)));
    }
    else
    {
        hpx::cout << (boost::format(
                    "Aggregate Writing Throughput: %1% [MB/s]\n") %
                (localities.size() * test_info.procs * test_info.wfiles *
                 test_info.count * test_info.bufsiz / tt / (1024*1024)));
        hpx::cout << (boost::format(
                    "\t Max Throughput per thread: %1% [MB/s]\n") %
                (test_info.wfiles * test_info.count * test_info.bufsiz /
                 min_time / (1024*1024)));
        hpx::cout << (boost::format("\t Min Throughput per thread: %1% [MB/s]\n") %
                (test_info.wfiles * test_info.count * test_info.bufsiz /
                 max_time / (1024*1024)));
    }

    hpx::cout<<hpx::endl;

}


void run_pxfs_file_test(test_info_type const& test_info)
{
    // Find the localities connected to this application.
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    std::vector<hpx::lcos::future<RESULT> > futures;
    futures.reserve(test_info.procs * localities.size());

    std::vector<RESULT> result_vector;
    result_vector.reserve(test_info.procs * localities.size());
    double min_time = -1.0, max_time = -1.0;

    // Keep track of the time required to execute.
    high_resolution_timer t;

    BOOST_FOREACH(hpx::naming::id_type const& node, localities)
    {
        // Initiate an asynchronous IO operation wait for it to complete without
        // blocking any of the HPX thread-manager threads.
        for(ssize_t proc=0; proc < test_info.procs; proc++)
        {
            futures.push_back(hpx::async<pxfs_file_test_action>
                    (node, test_info, proc));
        }
    }

    hpx::lcos::local::spinlock mtx;
    hpx::lcos::wait_each(
            hpx::util::unwrapped([&](RESULT r) {
                hpx::lcos::local::spinlock::scoped_lock lk(mtx);
                result_vector.push_back(r);
                }),
            futures);

    // overall performance
    double tt = t.elapsed();

    hpx::cout << "pxfs_file performance results:" << hpx::endl;

    if(test_info.rfiles > 0)
    {
        hpx::cout << (boost::format("%1% localities each has %2% threads,"
                    " Reading %3% files") % localities.size() %
                test_info.procs % test_info.rfiles);
    }
    else
    {
        hpx::cout << (boost::format("%1% localities each has %2% threads,"
                    " Writing %3% files") % localities.size() %
                    test_info.procs % test_info.wfiles);
    }

    char const* fmt = " with count %1% x buffer size %2%M: \n";
    hpx::cout << (boost::format(fmt) % test_info.count %
            (test_info.bufsiz * 1.0 / (1024 * 1024)));

    for(uint64_t idx=0; idx < (localities.size() * test_info.procs); idx++)
    {
        int loc = idx / test_info.procs;
        int pr = idx % test_info.procs;

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

    if(test_info.rfiles > 0)
    {
        hpx::cout << (boost::format(
                    "Aggregate Reading Throughput: %1% [MB/s]\n") %
                (localities.size() * test_info.procs * test_info.rfiles *
                 test_info.count * test_info.bufsiz / tt / (1024*1024)));
        hpx::cout << (boost::format(
                    "\t Max Throughput per thread: %1% [MB/s]\n") %
                (test_info.rfiles * test_info.count * test_info.bufsiz /
                 min_time / (1024*1024)));
        hpx::cout << (boost::format(
                    "\t Min Throughput per thread: %1% [MB/s]\n") %
                (test_info.rfiles * test_info.count * test_info.bufsiz /
                 max_time / (1024*1024)));
    }
    else
    {
        hpx::cout << (boost::format(
                    "Aggregate Writing Throughput: %1% [MB/s]\n") %
                (localities.size() * test_info.procs * test_info.wfiles *
                 test_info.count * test_info.bufsiz / tt / (1024*1024)));
        hpx::cout << (boost::format(
                    "\t Max Throughput per thread: %1% [MB/s]\n") %
                (test_info.wfiles * test_info.count * test_info.bufsiz /
                 min_time / (1024*1024)));
        hpx::cout << (boost::format("\t Min Throughput per thread: %1% [MB/s]\n") %
                (test_info.wfiles * test_info.count * test_info.bufsiz /
                 max_time / (1024*1024)));
    }

    hpx::cout<<hpx::endl;

}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    // extract command line argument
    boost::uint64_t rfiles = vm["rfiles"].as<boost::uint64_t>();
    boost::uint64_t wfiles = vm["wfiles"].as<boost::uint64_t>();
    ssize_t      bufsiz = vm["bufsiz"].as<ssize_t>();
    int      count = vm["count"].as<int>();
    size_t      procs = vm["procs"].as<size_t>();
    bool is_remove = vm.count("remove");
    bool has_orangefs = vm.count("orangefs");
    bool has_pxfs = vm.count("pxfs");
    std::string path;
    bool argument_error = false;

    if(procs > MAXPROCS)
    {
        hpx::cerr << "too many thread numbers!" << hpx::endl;
        argument_error = true;
    }

    if(rfiles == 0 && wfiles == 0)
    {
        hpx::cerr << "need to specify either to read or write files" << hpx::endl;
        argument_error = true;
    }

    if(rfiles > 0 && wfiles > 0)
    {
        hpx::cerr << "Can't read and write in the same test" << hpx::endl;
        argument_error = true;
    }

    if (vm.count("path")) {
        path = vm["path"].as<std::string>();
    } else {
        hpx::cerr << "Need to specify test path!!" << hpx::endl;
        argument_error = true;
    }

    if (!has_orangefs && !has_pxfs) {
        hpx::cerr << "Need to specify which file_class(es) to use: "
            "--orangefs, --pxfs!!" << hpx::endl;
        argument_error = true;
    }

    if (argument_error) {
        return hpx::finalize();
    }

    hpx::cout << "HPX disk performance benchmark with orangefs_file/pxfs_file "
        "async APIs." << hpx::endl;

    // init test_info
    test_info_type test_info(rfiles, wfiles, bufsiz, count,
            path, is_remove, procs);

    if (has_orangefs) {
        run_orangefs_file_test(test_info);
    }

    if (has_pxfs) {
        run_pxfs_file_test(test_info);
    }

    if(test_info.is_remove)
    {
        hpx::cout<<"removing test files ...\n";

        char filename[1024];
        uint64_t fileno = (test_info.rfiles > 0) ?
            test_info.rfiles : test_info.wfiles;

        for(uint32_t loc = 0; loc < hpx::find_all_localities().size(); ++loc)
        {
            for(ssize_t proc = 0; proc < test_info.procs; ++proc)
            {
                for(uint64_t i = 0; i < fileno; ++i)
                {
                    sprintf(filename, "%s/loc_%d_file%ld.%ld",
                            test_info.path.c_str(), loc, proc, i);
                    pvfs_unlink(filename);
                }
            }
        }

        hpx::cout<<hpx::endl;
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
        ( "procs" , value<size_t>()->default_value(1),
            "number of threads used for processing")
        ( "path" , value<std::string>(),
            "file path to place the testing files.")
        ( "orangefs" , "Testing I/O performance using orangefs_file class.")
        ( "pxfs" , "Testing I/O performance using pxfs_file class.")
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

