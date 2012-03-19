.. _ini_env_var_syntax:

***********************************
 INI Environmental Variable Syntax
***********************************

.. sectionauthor:: Hartmut Kaiser, Bryce Lelbach 

In HPX ini files, the following syntax:::

  ${FOO:default}

Will use the environmental variable FOO if it is set and default otherwise. 
No default has to be specified. Therefore this:::

  ${FOO}

refers to the environmental variable FOO. If FOO is not set or empty the 
overall expression will evaluate to an empty string.

The syntax:::

  $[section.key:default]

refers to the value held by the 'key' in the ini 'section' if it exists and 
default otherwise. No default has to be specified. Therefore this:::

  $[section.key]

refers to the 'key' inside the given 'section'. If the key is not set or 
empty the overall expression will evaluate to an empty string.

