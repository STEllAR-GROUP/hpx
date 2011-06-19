/* Boost bigint_storage_fixed.hpp header file
 *
 * Copyright 2007 Arseny Kapoulkine
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or
 * copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_BIGINT_BIGINT_STORAGE_FIXED_HPP
#define BOOST_BIGINT_BIGINT_STORAGE_FIXED_HPP

#include <vector>

namespace boost { namespace detail {
template <size_t N> struct bigint_storage_fixed
{
	template <typename T> class type
	{
		T data[N / sizeof(T)];
		size_t count;
	
	    size_t _max_size()
	    {
	    	return (N / sizeof(T));
	    }

	public:
	    type(): count(0)
	    {
	    }
		
		void resize(size_t size)
		{
			if (size > _max_size()) throw std::bad_alloc();
			
			count = size;
		}

		size_t size() const
		{
			return count;
		}

		bool empty() const
		{
			return count == 0;
		}

		const T* begin() const
		{
			return data;
		}
		
		T* begin()
		{
			return data;
		}

		const T* end() const
		{
			return data + count;
		}
		
		T* end()
		{
			return data + count;
		}

		const T& operator[](size_t index) const
		{
			BOOST_ASSERT(index < count);
			return data[index];
		}

		T& operator[](size_t index)
		{
			BOOST_ASSERT(index < count);
			return data[index];
		}
	
		void push_back(const T& value)
		{
			if (count >= _max_size()) throw std::bad_alloc();
			data[count++] = value;
		}

		void pop_back()
		{
			BOOST_ASSERT(count != 0);
			--count;
		}

		const T& front() const
		{
			BOOST_ASSERT(count != 0);
			return data[0];
		}
	
		T& front()
		{
			BOOST_ASSERT(count != 0);
			return data[0];
		}
	
		const T& back() const
		{
			BOOST_ASSERT(count != 0);
			return data[count - 1];
		}
		
		T& back()
		{
			BOOST_ASSERT(count != 0);
			return data[count - 1];
		}
	};
};
} }  // namespace boost::detail

#endif // BOOST_BIGINT_BIGINT_STORAGE_FIXED_HPP
