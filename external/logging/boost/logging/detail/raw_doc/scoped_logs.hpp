namespace boost { namespace logging {

/** 
@page scoped_logs Scoped Logs

- @ref scoped_logs_whatis 
- @ref scoped_logs_equivalence 
- @ref scoped_logs_context 
- @ref scoped_logs_fast 
- @ref scoepd_logs_multiple 


@section scoped_logs_whatis Scoped Logs - what happens?

The purpose of a "scoped log" is to log some information :
- at the beginning of a given @c scope
- at the end of the @c scope

The information is logged like this:

@code
[prefix] start of [information]
...
...
[prefix]   end of [information]
@endcode

Example:

@code
[1] 05:29.51 [dbg] start of testing inout
[2] 05:29.51 [dbg]   end of testing inout
@endcode


@section scoped_logs_equivalence Equivalence in code

To make it even clearer, using a scoped log:

@code
void func() {
    BOOST_SCOPED_LOG_CTX(LDBG) <<  "func()" ;
    // extra code
}
@endcode

Is equivalent with:

@code
void func() {
    LDBG <<  "start of func()" ;
    // extra code
    LDBG <<  "  end of func()" ;
}
@endcode

... of couse, using @c BOOST_SCOPED_LOG will have the right functionality even in the presence of exceptions.


Note that I encountered a very big problem, when implementing scoped logs: I don't know how you @ref workflow_2a "gather your message", when using the logs. 
In other words, I don't know your Usage Syntax. So I had to make a few assumptions, as you'll see.



@section scoped_logs_context The easy way - BOOST_SCOPED_LOG_CTX

This allows you to simply log context in a straighforward manner, using the operator << ; context includes :
- any variable in the local scope
- any parameter passed to your function

Example:

@code
#define LDBG BOOST_LOG_USE_LOG_IF_LEVEL(g_l(), g_log_filter(), debug ) 
...

void func(int a, const char * str) {
    BOOST_SCOPED_LOG_CTX(LDBG) << "func(" << a << ", str=" << str << ")";
    ...
}
@endcode

Things you should know:
- When using @c BOOST_SCOPED_LOG_CTX, you pass as parameter, one of the @ref macros_use "macros" you've already defined.
- When using @c BOOST_SCOPED_LOG_CTX, you'll always @em have to use @c << to write the message, even though
  your logger might use a different syntax (see @ref workflow_2a "gathering the message")
- When you use this macro (BOOST_SCOPED_LOG_CTX), a temporary variable is created, which will hold the result
  of gathering your context. In the above case, this variable will contain the contents of: \n
  <tt> "func(" << a << ", str=" << str << ")"; </tt>
- @c BOOST_SCOPED_LOG_CTX preserves the "is_enabled" policy of the underlying logger. In other words, if you do \n
  <tt> BOOST_SCOPED_LOG_CTX(LDBG) << "func" << some_time_consuming_func(); </tt> \n
  and the logger is disabled, the @c some_time_consuming_func() will not be called




@section scoped_logs_fast The fast way - BOOST_SCOPED_LOG

The fast way makes no assumptions about your @ref workflow_2a "Usage Syntax". However, it's very limited in use:
- You can only specify a string literal
- If you use some operator (like "<<") when logging, you'll have to specify it as first argument
- You cannot use any variable from your scope, nor any other variables (in fact, this is implied by the first item)

Example:

@code
#define LDBG BOOST_LOG_USE_LOG_IF_LEVEL(g_l(), g_log_filter(), debug ) 
...
void func(int a, const char * str) {
    BOOST_SCOPED_LOG(LDBG << , "testing inout" );
    ...
}
@endcode

It's fast, because:
- It uses no extra temporary variable
- It contactenates "start of " + message, and "end of " + message at <tt>compile time</tt>




@section scoepd_logs_multiple Multiple scoped logs

...are allowed. You can create a @c BOOST_SCOPED_LOG or @c BOOST_SCOPED_LOG_CTX at any time - within the body of a function, with the only limitation
that you can't have 2 on the same line.

Example:
@code
void func(int a, const char * str) {
    BOOST_SCOPED_LOG_CTX(LDBG) << "func(" << a << ", str=" << str << ")";
    int i = 0;
    BOOST_SCOPED_LOG_CTX(LDBG) << "i =" << i;
}
@endcode

*/

}}
