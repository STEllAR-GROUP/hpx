/*****************************************************

    Copyright (c) 1999 Junling Ma

	File : parse.h
   Author : Junling Ma
   Date : October 21, 1999

   Overview:

	This file contains the definations needed to use the
   parameter file parser.

	A parameter file contains defination of parameters like:
   	Name Separator Value

   where
   	Name is a string that doesn't contain a separator,
      Separator can be any charactor, such as = , : and so on.

   For example,
   	First = 1
      Second = 2

   Useful functions are defined below.
   Usaully you parse a file by calling Parse, and after you get
   the parameters list, you call GetInt or other Get functions
   to retreat the parameter values. At last you call a DeleteRecord
   to release the parameter list you got from Parse.

*****************************************************/

/* the struct Record defines the list of parameter records*/
typedef struct _Record {
	char *Name;    /* the name of the parameter */
   char *Value;   /* the value of the parameter */
	struct _Record *Next;   /* next record */
} Record;


/******************************************************

	Function DeleteRecord deletes the give record list.
	Parameters:
   	list: the list to delete.

******************************************************/
void DeleteRecord(Record *list);

/*************************************************

	Function Parse parse the parameter file and get the parameter list.
   Returns the parameter list.
   Parameters:
   	FileName: the file to parse;
      Separator: the separator between the Name and Value

*************************************************/
Record *Parse(char *FileName, char Separator);

/*************************************************

	Function GetInt get the int value of a parameter.
   Returns 0 if fail to find the parameter, 1 if succeed.
   Parameters:
   	List: the list to search;
      Name: the name of the parameter;
      Res: the pointer a int to store the value.

*************************************************/
int GetInt(Record *List, char *Name, int *Res);

/*************************************************

	Function GetFloat get the float value of a parameter.
   Returns 0 if fail to find the parameter, 1 if succeed.
   Parameters:
		List: the list to search;
      Name: the name of the parameter;
      Res: the pointer a float to store the value.

*************************************************/
int GetFloat(Record *List, char *Name, float *Res);
int GetDouble(Record *List, char *Name, double *Res);

/*************************************************

	Function GetBOOL get the BOOL value of a parameter.
   Returns 0 if fail to find the parameter, 1 if succeed.
   Parameters:
   	List: the list to search;
		Name: the name of the parameter;
		Res: the pointer a BOOL to store the value.

*************************************************/
int GetBOOL(Record *List, char *Name, int *Res);

/*************************************************

	Function GetString get the string value of a parameter.
   Returns the value, or NULL if fail to find the parameter.
   Parameters:
   	List: the list to search;
      Name: the name of the parameter;

*************************************************/
char *GetString(Record *List, char *Name);
