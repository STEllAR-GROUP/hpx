
#if !defined(HPX_Sg7J0M91KTJ7FTMezA1gNjwaA2LCgq70PgnGNvX3)
#define HPX_Sg7J0M91KTJ7FTMezA1gNjwaA2LCgq70PgnGNvX3

#include <hpx/hpx_fwd.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_uint.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/io.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>

namespace hpx { namespace performance_counters { namespace server
{

    namespace qi = boost::spirit::qi;
    namespace ascii = boost::spirit::ascii;

    struct proc_statm
    {
        boost::uint64_t size;
        boost::uint64_t resident;
        boost::uint64_t share;
        boost::uint64_t text;
        boost::uint64_t lib;
        boost::uint64_t data;
        boost::uint64_t dt;   
    };

}}}

BOOST_FUSION_ADAPT_STRUCT(                                                       
    hpx::performance_counters::server::proc_statm,                               
    (boost::uint64_t, size)                                                      
    (boost::uint64_t, resident)                                                  
    (boost::uint64_t, share)                                                     
    (boost::uint64_t, text)                                                      
    (boost::uint64_t, lib)                                                       
    (boost::uint64_t, data)                                                      
    (boost::uint64_t, dt)                                                        
)

namespace hpx { namespace performance_counters { namespace server                
{                                                                                
                                                                                 
    template <typename Iterator>                                                 
    struct proc_statm_parser : qi::grammar<Iterator, proc_statm(), ascii::space_type>
    {                                                                            
        proc_statm_parser() : proc_statm_parser::base_type(start)                
        {                                                                        
            using qi::ulong_;           // does not work?                        
                                                                                 
            start = ulong_                                                       
                >> ulong_                                                        
                >> ulong_                                                        
                >> ulong_                                                        
                >> ulong_                                                        
                >> ulong_                                                        
                >> ulong_                                                        
                ;                                                                
        }                                                                        
                                                                                 
        qi::rule<Iterator, proc_statm(), ascii::space_type> start;               
    };                                                                           
                                                                                 
}}}

namespace hpx { namespace performance_counters { namespace server
{       
        typedef hpx::performance_counters::server::proc_statm proc_statm_type; 
        // returns virtual memory value        
        boost::uint64_t read_psm_vm()                                       
        {                                                                        
            using boost::spirit::ascii::space;                                   
            typedef std::string::const_iterator iterator_type;                   
            typedef hpx::performance_counters::server::proc_statm_parser<iterator_type>
                proc_statm_parser;                                               
                                                                                 
            proc_statm_parser psg;                                               
            std::string in_string;   
            boost::uint32_t pid = getpid();
            std::string filename = boost::str(boost::format("/proc/%1%/statm") % pid);
            std::stringstream buffer;                                            
            std::ifstream infile(filename.c_str());                              
            buffer << infile.rdbuf();                                            
            in_string = buffer.str();                                            
            hpx::performance_counters::server::proc_statm psm;                   
            iterator_type itr = in_string.begin(); 
            iterator_type end = in_string.end(); 
            bool r = phrase_parse(itr, end, psg, space, psm);                    
            if(r)                                                                
                std::cout << "parsing success" << std::endl;
            else                                                                 
            {                                                                    
                std::cout << "parsing failure" << std::endl;                     
            }                                                                    
            return psm.size;
        }
        
        // returns resident memory value        
        boost::uint64_t read_psm_resident()                                       
        {                                                                        
            using boost::spirit::ascii::space;                                   
            typedef std::string::const_iterator iterator_type;                   
            typedef hpx::performance_counters::server::proc_statm_parser<iterator_type>
                proc_statm_parser;                                               
                                                                                 
            proc_statm_parser psg;                                               
            std::string in_string;   
            boost::uint32_t pid = getpid();
            std::string filename = boost::str(boost::format("/proc/%1%/statm") % pid);
            std::stringstream buffer;                                            
            std::ifstream infile(filename.c_str());                              
            buffer << infile.rdbuf();                                            
            in_string = buffer.str();                                            
            hpx::performance_counters::server::proc_statm psm;                   
            iterator_type itr = in_string.begin(); 
            iterator_type end = in_string.end(); 
            bool r = phrase_parse(itr, end, psg, space, psm);                    
            if(r)                                                                
                std::cout << "parsing success" << std::endl;
            else                                                                 
            {                                                                    
                std::cout << "parsing failure" << std::endl;                     
            }                                                                    
            return psm.resident;
        }
}}}
#endif //HPX_Sg7J0M91KTJ7FTMezA1gNjwaA2LCgq70PgnGNvX3
