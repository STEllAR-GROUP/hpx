////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// disk performance benchmark using OrangeFS direct interface or c++ file I/O API

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sys/times.h>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/shared_array.hpp>
#include <boost/thread.hpp>
#include <boost/program_options.hpp>

#include <hpx/util/high_resolution_timer.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::util::high_resolution_timer;

//////////////////////////////////////////////////////////////////

#define MAXPROCS 1000

static boost::uint64_t      bufsiz = 10 * 1024;
static int      count = 1;
static int      procs = 1;
static bool     is_ofsio = false;
static bool     is_remove = false;

static std::string filepath;

struct RESULT {
  double	real;
  double	user;
  double	sys;
};

/* ------------------------  added pvfs header stuff --------------- */

#ifdef __cplusplus
extern "C" {
#endif

#include <pvfs2-usrint.h>
#include <pvfs2.h>

#ifdef __cplusplus
} //extern "C" {
#endif

/* -------------------------  end pvfs header stuff --------------- */





///////////////////////////////////////////////////////////////////////////////
void do_write_files_test(boost::uint64_t wfiles, int proc, RESULT &r)
{
    // perform the I/O operations here
   char filename[1024];
   clock_t	start;
   clock_t	end;
   struct tms	t1;
   struct tms	t2;
//   RESULT r;
   std::ofstream file;
   char     *buf;

   if((buf = (char *)valloc(bufsiz)) == NULL)
   {
       std::cerr << "error when create char array." << std::endl;
       return;
   }

   srand((unsigned)time(0));

   start = times(&t1);

   for(boost::uint64_t i=0; i<wfiles; i++)
   {
       if(is_ofsio)
       {
           // using OrageFS syscalls
           int	fd;
           int	oflags;

           oflags = O_WRONLY|O_CREAT;
           sprintf(filename, "%s/file%d.%ld", filepath.c_str(), proc, i);

           if ((fd = pvfs_open(filename, oflags, 0777)) == -1) {
               std::cout<<"Unable to open orangeFS file "<<filename<<std::endl;
               return;
           }

           for (int j = 0; j < count; j++) {
               for(boost::uint64_t c = 0; c < bufsiz; c++)
                   buf[c] = (unsigned int) rand() % 256;

               if (pvfs_write(fd,buf,bufsiz) <= 0) {
                   std::cout<<"Unable to write orangeFS file "<<
                       filename<<std::endl;
                   return;
               }
           }

           pvfs_close(fd);
       }
       else
       {
           sprintf(filename, "%s/file%d.%ld", filepath.c_str(), proc, i);
           file.open(filename);
           if(file.is_open())
           {
               for(boost::uint64_t c = 0; c < bufsiz; c++)
                   buf[c] = (unsigned int) rand() % 256;

               for(int j=0; j<count; j++)
               {
                   file.write(buf, bufsiz);
               }
           }
           else
           {
               std::cout<<"Unable to write to "<<filename<<std::endl;
           }
           file.close();
       }
   }

   end = times(&t2);
   r.real = ((double) end - (double) start) / (double) sysconf(_SC_CLK_TCK);
   r.user = ((double) t2.tms_utime - (double) t1.tms_utime) / (double) sysconf(_SC_CLK_TCK);
   r.sys = ((double) t2.tms_stime - (double) t1.tms_stime) / (double) sysconf(_SC_CLK_TCK);

    free(buf);
}


///////////////////////////////////////////////////////////////////////////////
// this function will be executed by a dedicated OS thread
void do_read_files_test(boost::uint64_t rfiles, int proc, RESULT &r)
{
    // perform the I/O operations here
   char filename[1024];
   clock_t	start;
   clock_t	end;
   struct tms	t1;
   struct tms	t2;
   std::ifstream file;
   char     *buf;

   if((buf = (char *)valloc(bufsiz)) == NULL)
   {
       std::cerr << "error when create char array." << std::endl;
       return;
   }

   start = times(&t1);

   // buf should already been allocated
   for(boost::uint64_t i=0; i<rfiles; i++)
   {
       if(is_ofsio)
       {
           // using OrageFS syscalls
           int	fd;
           int	oflags;

           oflags = O_RDONLY;
           sprintf(filename, "%s/file%d.%ld", filepath.c_str(), proc, i);

           if ((fd = pvfs_open(filename, oflags)) == -1) {
               std::cout<<"Unable to open orangeFS file "<<filename<<std::endl;
               return;
           }

           for (int j = 0; j < count; j++) {
                   if (pvfs_read(fd,buf,bufsiz) <= 0) {
                       std::cout<<"Unable to read orangeFS file "<<
                           filename<<std::endl;
                       return;
                   }
           }

           pvfs_close(fd);
       }
       else
       {
           sprintf(filename, "%s/file%d.%ld", filepath.c_str(), proc, i);
           file.open(filename);
           if(file.is_open())
           {
               for(int j=0; j<count; j++)
               {
                   file.read(buf, bufsiz);
               }
           }
           else
           {
               std::cout<<"Unable to read from "<<filename<<std::endl;
           }
           file.close();
       }
   }

   end = times(&t2);
   r.real = ((double) end - (double) start) / (double) sysconf(_SC_CLK_TCK);
   r.user = ((double) t2.tms_utime - (double) t1.tms_utime) / (double) sysconf(_SC_CLK_TCK);
   r.sys = ((double) t2.tms_stime - (double) t1.tms_stime) / (double) sysconf(_SC_CLK_TCK);

   free(buf);
}


///////////////////////////////////////////////////////////////////////////////
int now_run(variables_map& vm)
{
    // extract command line argument
    boost::uint64_t rfiles = vm["rfiles"].as<boost::uint64_t>();
    boost::uint64_t wfiles = vm["wfiles"].as<boost::uint64_t>();
    is_remove = vm.count("remove");
    is_ofsio = vm.count("orangefs");

    if (vm.count("path"))
    {
        filepath = vm["path"].as<std::string>();
    } else
    {
        std::cerr << "Need to specify test path!!" << std::endl;
        return -1;
    }

    if(procs > MAXPROCS)
    {
        std::cerr << "too many proc numbers!" << std::endl;
        return -1;
    }

    if(rfiles == 0 && wfiles == 0)
    {
        std::cerr << "need to specify either to read or write files" << std::endl;
        return -1;
    }

    if(rfiles > 0 && wfiles > 0)
    {
        std::cerr << "Can't read and write in the same test" << std::endl;
        return -1;
    }

    std::cout<<"C++ synchronous disk performance test with local file system "
        "and orangefs file system."<<std::endl;

    if(is_ofsio)
    {
        std::cout<<"Using OrageFS path: "<<filepath<<std::endl;
    }
    else
    {
        std::cout<<"Using local path: "<<filepath<<std::endl;
    }

    {
        // Keep track of the time required to execute.
        high_resolution_timer t;
        boost::shared_array<RESULT> r_ptr (new RESULT[procs]);
        double min_time = -1.0, max_time = -1.0;

        std::vector<boost::thread> thr;

        if(rfiles > 0) // read tests
        {
            // Initiate an asynchronous IO operation wait for it to complete without
            // blocking any of the HPX thread-manager threads.
            for(int proc=0; proc < procs; proc++)
            {
                thr.push_back(boost::thread(do_read_files_test, rfiles,
                            proc, r_ptr[proc]));
            }
        }
        else
        {

            for(int proc=0; proc < procs; proc++)
            {
                thr.push_back(boost::thread(do_write_files_test, wfiles,
                            proc, r_ptr[proc]));
            }
        }

        for(int proc=0; proc < procs; proc++)
        {
            thr[proc].join();
        }

        // overall performance 
        double tt = t.elapsed();

        if (rfiles > 0)
        {
            std::cout << (boost::format("Reading %1% files") % rfiles);
        } else
        {
            std::cout << (boost::format("Writing %1% files") % wfiles);
        }

        char const* fmt = " with count %1% x buffer size %2%M and %3% procs: \n";
        std::cout << (boost::format(fmt) % count % (bufsiz*1.0 /(1024*1024)) % procs);

        for(int proc=0; proc < procs; proc++)
        {
            std::cout << (boost::format("process %1% time: Real %2% [s], User %3% [s], System %4% [s]\n") % proc % r_ptr[proc].real % r_ptr[proc].user % r_ptr[proc].sys);
            if ((min_time < 0) || (r_ptr[proc].real < min_time) )
            {
                min_time = r_ptr[proc].real;
            }
            if ((max_time < 0) || (r_ptr[proc].real > max_time) )
            {
                max_time = r_ptr[proc].real;
            }
        }

        std::cout<<"-------------------------------------------\n";
        std::cout<<(boost::format("Total elapsed time: %1% [s]\n") % tt);
        std::cout<<(boost::format("Aggregate %1% Throughput: %2% [MB/s]\n") % 
                ((rfiles>0) ? "Reading" : "Writing") %
                (procs * count * bufsiz / tt / (1024*1024)));
        std::cout<<(boost::format("\t Max Throughput per thread: %1% [MB/s]\n") % 
                (count * bufsiz / min_time / (1024*1024)));
        std::cout<<(boost::format("\t Min Throughput per thread: %1% [MB/s]\n") % 
                (count * bufsiz / max_time / (1024*1024)));

        std::cout<<std::endl<<std::endl;

        if (is_remove)
        {
            char filename[1024];
            boost::uint64_t fileno = (rfiles>0) ? rfiles : wfiles;

            for(int proc=0; proc < procs; proc++)
            {
                for(boost::uint64_t i=0; i<fileno; i++)
                {
                    sprintf(filename, "%s/file%d.%ld",
                            filepath.c_str(), proc, i);
                    if (is_ofsio)
                    {
                        pvfs_unlink(filename);
                    } else {
                        remove(filename);
                    }
                }
            }
        }
    }

    return 0;
}



///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: [options]");

    desc_commandline.add_options()
        ( "rfiles" , value<boost::uint64_t>()->default_value(0),
            "number of files to for reading")
        ( "wfiles" , value<boost::uint64_t>()->default_value(0),
            "number of files to for writing")
        ( "bufsiz" , value<boost::uint64_t>(&bufsiz)->default_value(10*1024),
            "buffer size (in bytes)")
        ( "path" , value<std::string>(),
            "file test path, default is the current directory.")
        ( "count", value<int>(&count)->default_value(1),
            "number of bufsiz in a file")
        ( "procs" , value<int>(&procs)->default_value(1),
            "number of threads used for processing")
        ( "orangefs" , "use OrangeFS file path.")
        ( "remove" , "remove test files.")
        ( "help" , "print help message.")
        ;

        variables_map vm;
        store(parse_command_line(argc, argv, desc_commandline), vm);
        notify(vm);

        if (vm.count("help"))
        {
            std::cerr << desc_commandline << std::endl;
            return 0;
        }

        return now_run(vm);

}

