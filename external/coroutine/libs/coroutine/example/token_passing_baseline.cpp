//  Copyright (c) 2006, Giovanni P. Deretta
//
//  This code may be used under either of the following two licences:
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy 
//  of this software and associated documentation files (the "Software"), to deal 
//  in the Software without restriction, including without limitation the rights 
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
//  copies of the Software, and to permit persons to whom the Software is 
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in 
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
//  THE SOFTWARE. OF SUCH DAMAGE.
//
//  Or:
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <vector>
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/bind.hpp>


typedef boost::asio::ip::tcp::acceptor acceptor_type;
typedef boost::asio::ip::tcp::endpoint endpoint_type;
typedef boost::optional<boost::asio::error> error_type;

boost::asio::io_service demuxer;

using boost::asio::placeholders::bytes_transferred;
using boost::asio::placeholders::error;

void frob(char * begin, size_t len) {
  char first = begin[0];
  for(size_t i=0; i<len -1; i++)
    begin[i] ^=begin[i+1];
  begin[len-1] ^= first;
}

class thread {
public:
  thread(acceptor_type* acceptor,
	 endpoint_type* endpoint,
	 int index, 
	 int counter, 
	 int token_size) :
    m_sink(acceptor->io_service()),
    m_source(acceptor->io_service()),
    m_index(index),
    m_counter(counter),
    m_token(new char[token_size]),
    m_token_size(token_size){
    for(int i=0; i<token_size; i++) m_token[i] = 0;
    //std::cout <<"[Thread "<<m_index<<"]: Accept or connect in progress...\n";
    acceptor->async_accept(m_sink, 
			   boost::bind(&thread::ready, 
				       this, 
				       &m_accept_error, 
				       error));
    
    m_source.async_connect(*endpoint, 
			   boost::bind(&thread::ready, 
				       this, 
				       &m_connect_error, 
				       error));
  }

  void ready(error_type* to, error_type from) {
    *to = from;
    if(m_accept_error && m_connect_error) {
      if(*m_accept_error || *m_connect_error) {
	//std::cout <<"[Thread "<<m_index<<"]: Accept/connect error\n";
	m_source.io_service().interrupt();
      } else {

	//std::cout<<"[Thread "<<m_index<<"] :receiving token"<<std::endl;
	boost::asio::async_read(m_source,
				boost::asio::buffer(m_token, m_token_size),
				boost::bind(&thread::main, 
					    this,
					    &m_read_error, 
					    error, 
					    bytes_transferred));
	
	//std::cout<<"[Thread "<<m_index<<"] :sending token"<<std::endl;
	boost::asio::async_write(m_sink,
				 boost::asio::buffer(m_token, m_token_size),
				 boost::bind(&thread::main, 
					     this,
					     &m_write_error, 
					     error, 
					     bytes_transferred));
	
	//std::cout <<"[Thread "<<m_index<<"]: Accept and connect completed\n";
	main(&m_read_error, error_type(), size_t());
      }
    }
  }
    
  void main(error_type* to, error_type from, size_t) {
    *to = from;
    if(m_counter == 0) 
      m_source.io_service().interrupt();
      
    if(m_write_error) {
      if(*m_write_error) {
	//std::cout<<"[Thread "<<m_index<<"] :error while writing, exiting..."<<std::endl;
	m_source.io_service().interrupt();
      }
      //std::cout<<"[Thread "<<m_index<<"] :token sent"<<std::endl;
      m_write_error = error_type();  
	
      frob(m_token,m_token_size);

      //std::cout<<"[Thread "<<m_index<<"] :sending token"<<std::endl;
      boost::asio::async_write(m_sink,
			       boost::asio::buffer(m_token, m_token_size),
			       boost::bind(&thread::main, 
					   this,
					   &m_write_error, 
					   error, 
					   bytes_transferred));
      m_counter --;
    } 
    
    if(m_read_error) {
      if(*m_read_error) {
	//std::cout<<"[Thread "<<m_index<<"] :error while readin, exiting..."<<std::endl;
	m_source.io_service().interrupt();
      }
      //std::cout<<"[Thread "<<m_index<<"] :token read"<<std::endl;
      m_read_error = error_type();   

      //std::cout<<"[Thread "<<m_index<<"] :receiving token"<<std::endl;
      boost::asio::async_read(m_source,
			      boost::asio::buffer(m_token, m_token_size),
			      boost::bind(&thread::main, 
					  this,
					  &m_read_error, 
					  error, 
					  bytes_transferred));
      
    } 
  }
  
  boost::asio::ip::tcp::socket m_sink;
  boost::asio::ip::tcp::socket m_source;
  error_type m_accept_error;
  error_type m_connect_error;
  error_type m_read_error;
  error_type m_write_error;
  int m_index;
  int m_counter;
  char * m_token;
  int m_token_size;

};


typedef std::vector<boost::shared_ptr<acceptor_type> > acceptor_vector_type;
typedef std::vector< endpoint_type > endpoint_vector_type;
typedef std::vector<boost::shared_ptr<thread> > thread_vector_type;

int main(int argc, char** argv) {
  int count = ((argc >= 2) ? boost::lexical_cast<int>(argv[1]): 100);
  int count_2 = ((argc >= 3) ? boost::lexical_cast<int>(argv[2]): 1);
  int base_port = ((argc >= 4) ? boost::lexical_cast<int>(argv[3]): 30000);
  int token_size = ((argc == 5) ? boost::lexical_cast<int>(argv[4]): 4096);

  acceptor_vector_type acceptor_vector;
  endpoint_vector_type endpoint_vector;
  thread_vector_type thread_vector;
  acceptor_vector.reserve(count_2);
  endpoint_vector.reserve(count_2);
  thread_vector.reserve(count_2);
  for(int i= 0; i< count_2; i++) {
    endpoint_vector.push_back
      (endpoint_type
       (boost::asio::ip::address_v4::from_string("127.0.0.1"), 
	base_port + i));
    acceptor_vector.push_back
      (boost::shared_ptr<acceptor_type> 
       (new acceptor_type(demuxer)));
    acceptor_vector.back()->open(endpoint_vector.back().protocol());
    acceptor_vector.back()->set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    acceptor_vector.back()->bind(endpoint_vector.back());
    acceptor_vector.back()->listen();
  }

  thread_vector.push_back(boost::shared_ptr<thread>
			  (new thread(&*acceptor_vector.back(), 
				      &endpoint_vector.back(),
				      0,
				      count, 
				      token_size)));
  for(int i=1; i< count_2; i++) {
    thread_vector.push_back(boost::shared_ptr<thread>
			    (new thread(&*acceptor_vector.at(i-1), 
					&endpoint_vector.at(i-1),
					i,
					count,
					token_size)));
  }
  demuxer.run(); 
} 
 
 
