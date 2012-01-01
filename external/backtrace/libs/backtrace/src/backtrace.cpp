//
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2012 Hartmut Kaiser
//  Copyright (c) 2010 Artyom Beilis (Tonkikh)
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
#define BOOST_BACKTRACE_SOURCE

#include <boost/config.hpp>
#include <boost/backtrace.hpp>
#include <boost/lexical_cast.hpp>

#if defined(__linux) || defined(__APPLE__) || defined(__sun)
#define BOOST_HAVE_EXECINFO
#define BOOST_HAVE_DLADDR
#endif

#if defined(__GNUC__)
#define BOOST_HAVE_ABI_CXA_DEMANGLE
#endif

#ifdef BOOST_HAVE_EXECINFO
#include <execinfo.h>
#endif

#ifdef BOOST_HAVE_ABI_CXA_DEMANGLE
#include <cxxabi.h>
#endif

#ifdef BOOST_HAVE_DLADDR
#include <dlfcn.h>
#endif
#include <string.h>
#include <stdlib.h>
#include <ostream>
#include <sstream>
#include <iomanip>

#if defined(BOOST_MSVC)
#include <windows.h>
#include <winbase.h>
#include <stdlib.h>
#include <dbghelp.h>
#endif


namespace boost {

    namespace stack_trace {
        #if defined(BOOST_HAVE_EXECINFO)

        int trace(void **array,std::size_t n)
        {
            return :: backtrace(array,n);
        }

        #elif defined(BOOST_MSVC)

        int trace(void **array,std::size_t n)
        {
            if(n>=63)
                n=62;
            return RtlCaptureStackBackTrace(ULONG(0),ULONG(n),array,NULL);
        }

        #else

        int trace(void ** /*array*/,std::size_t /*n*/)
        {
            return 0;
        }

        #endif

        #if defined(BOOST_HAVE_DLADDR) && defined(BOOST_HAVE_ABI_CXA_DEMANGLE)

        std::string get_symbol(void *ptr)
        {
            if(!ptr)
                return std::string();
            std::ostringstream res;
            res.imbue(std::locale::classic());
            res << std::left << std::setw(sizeof(void*)*2) << std::setfill(' ') << ptr <<": ";
            Dl_info info = {0, 0, 0, 0};
            if(dladdr(ptr,&info) == 0) {
                res << "???";
            }
            else {
                if(info.dli_sname) {
                    int status = 0;
                    char *demangled = abi::__cxa_demangle(info.dli_sname,0,0,&status);
                    if(demangled) {
                        res << demangled;
                        free(demangled);
                    }
                    else {
                        res << info.dli_sname;
                    }
                }
                else {
                    res << "???";
                }

                unsigned offset = (char *)ptr - (char *)info.dli_saddr;
                res << std::hex <<" + 0x" << offset ;

                if(info.dli_fname)
                    res << " in " << info.dli_fname;
            }
           return res.str();
        }

        std::string get_symbols(void *const *addresses,std::size_t size)
        {
            std::string res = boost::lexical_cast<std::string>(size) + ((1==size)?" frame:":" frames:");
            for(std::size_t i=0;i<size;i++) {
                std::string tmp = get_symbol(addresses[i]);
                if(!tmp.empty()) {
                    res+='\n';
                    res+=tmp;
                }
            }
            return res;
        }
        void write_symbols(void *const *addresses,std::size_t size,std::ostream &out)
        {
            out << size << ((1==size)?" frame:":" frames:");
            for(std::size_t i=0;i<size;i++) {
                std::string tmp = get_symbol(addresses[i]);
                if(!tmp.empty()) {
                    out << '\n' << tmp;
                }
            }
            out << std::flush;
        }

        #elif defined(BOOST_HAVE_EXECINFO)
        std::string get_symbol(void *address)
        {
            char ** ptr = backtrace_symbols(&address,1);
            try {
                if(ptr == 0)
                    return std::string();
                std::string res = ptr[0];
                free(ptr);
                ptr = 0;
                return res;
            }
            catch(...) {
                free(ptr);
                throw;
            }
        }

        std::string get_symbols(void * const *address,std::size_t size)
        {
            char ** ptr = backtrace_symbols(address,size);
            try {
                if(ptr==0)
                    return std::string();
                std::string res = boost::lexical_cast<std::string>(size) + ((1==size)?" frame:":" frames:");
                for(std::size_t i=0;i<size;i++) {
                    res+='\n';
                    res+=ptr[i];
                }
                free(ptr);
                ptr = 0;
                return res;
            }
            catch(...) {
                free(ptr);
                throw;
            }
        }


        void write_symbols(void *const *addresses,std::size_t size,std::ostream &out)
        {
            char ** ptr = backtrace_symbols(addresses,size);
            out << size << ((1==size)?" frame:":" frames:");
            try {
                if(ptr==0)
                    return;
                for(int i=0;i<size;i++)
                    out << '\n' << ptr[i];
                free(ptr);
                ptr = 0;
                out << std::flush;
            }
            catch(...) {
                free(ptr);
                throw;
            }
        }

        #elif defined(BOOST_MSVC)

        namespace {
            HANDLE hProcess = 0;
            bool syms_ready = false;

            void init()
            {
                if(hProcess == 0) {
                    hProcess = GetCurrentProcess();
                    SymSetOptions(SYMOPT_DEFERRED_LOADS);

                    if (SymInitialize(hProcess, NULL, TRUE))
                    {
                        syms_ready = true;
                    }
                }
            }
        }

        std::string get_symbol(void *ptr)
        {
            if(ptr==0)
                return std::string();
            init();
            std::ostringstream ss;
            ss << std::left << std::setw(sizeof(void*)*2) << std::setfill(' ') << ptr;
            if(syms_ready) {
                DWORD64  dwDisplacement = 0;
                DWORD64  dwAddress = (DWORD64)ptr;

                std::vector<char> buffer(sizeof(SYMBOL_INFO) + MAX_SYM_NAME);
                PSYMBOL_INFO pSymbol = (PSYMBOL_INFO)&buffer.front();

                pSymbol->SizeOfStruct = sizeof(SYMBOL_INFO);
                pSymbol->MaxNameLen = MAX_SYM_NAME;

                if (SymFromAddr(hProcess, dwAddress, &dwDisplacement, pSymbol))
                {
                    ss <<": " << pSymbol->Name << std::hex << " +0x" << dwDisplacement;
                }
                else
                {
                    ss << ": ???";
                }
            }
            return ss.str();
        }

        std::string get_symbols(void *const *addresses,std::size_t size)
        {
            std::string res = boost::lexical_cast<std::string>(size) + ((1==size)?" frame:":" frames:");
            for(std::size_t i=0;i<size;i++) {
                std::string tmp = get_symbol(addresses[i]);
                if(!tmp.empty()) {
                    res+='\n';
                    res+=tmp;
                }
            }
            return res;
        }
        void write_symbols(void *const *addresses,std::size_t size,std::ostream &out)
        {
            out << size << ((1==size)?" frame:":" frames:");
            for(std::size_t i=0;i<size;i++) {
                std::string tmp = get_symbol(addresses[i]);
                if(!tmp.empty()) {
                    out << '\n' << tmp;
                }
            }
            out << std::flush;
        }

        #else

        std::string get_symbol(void *ptr)
        {
            if(!ptr)
                return std::string();
            std::ostringstream res;
            res.imbue(std::locale::classic());
            res << std::left << std::setw(sizeof(void*)*2) << std::setfill(' ') << ptr;
            return res.str();
        }

        std::string get_symbols(void *const *ptrs,std::size_t size)
        {
            if(!ptrs)
                return std::string();
            std::ostringstream res;
            res.imbue(std::locale::classic());
            write_symbols(ptrs,size,res);
            return res.str();
        }

        void write_symbols(void *const *addresses,std::size_t size,std::ostream &out)
        {
            out << size << ((1==size)?" frame:":" frames:");
            for(std::size_t i=0;i<size;i++) {
                if(addresses[i]!=0)
                    out << '\n' << std::left << std::setw(sizeof(void*)*2) << std::setfill(' ') << addresses[i];
            }
            out << std::flush;
        }

        #endif

    } // stack_trace

} // boost

