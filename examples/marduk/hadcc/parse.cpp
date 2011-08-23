/*****************************************************

    Copyright (c) 1999 Junling Ma

    File : parse.c
    Author : Junling Ma
    Date : October 21, 1999

    Overview:

    This file contains the implementation of a parameter
    file parser.

    Please refer to file parse.h and function main in this
    file for the usage.

    to test it, define a symbol "Test" or uncomment the #define
    line below. Then compile it.
    Usage of the test:

        Parse File Separator Parameter1 Parameter 2 ...

    where
        File is the parameter file to parse;
        Separator is the separator between the parameter name and value;
        Parameter1 ... are the parameters you want to display

*****************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "parse.h"

/******************************************************

    Function SearchRecord search the list for a record with gavin name.
    Returns the value or NULL if not found.
    Parameters:
        Name: the parameter name to search.
******************************************************/
char *SearchRecord(Record *List, char *Name)
{
    char name[1000];
    int i=0;

    if (Name==NULL || List==NULL) return NULL;

    while (Name[i]!=0) {
        name[i]=toupper(Name[i]);
        i++;
    }
    name[i]=0;
    while (List!=NULL && strcmp(name,List->Name)) List=List->Next;
    if (List==NULL)
        return NULL;
    else
        return List->Value;
}

/******************************************************

    Function DeleteRecord deletes the give record list.
    Parameters:
        list: the list to delete.

******************************************************/
void DeleteRecord(Record *list)
{
    if (list!=NULL) {
        DeleteRecord(list->Next);
        if (list->Name!=NULL)
            free(list->Name);
        if (list->Value!=NULL)
            free(list->Value);
        free(list);
    }
}

/*******************************************************

    Function PutRecord add a record to the list.
    Returns the updated list, or NULL if fail.
    Parameters:
        List: the list to add;
        Name: the name of the field;
        Value: the value;

*******************************************************/
Record *PutRecord(Record *List, char *Name, char *Value)
{
    /*  return for invalid parameters */
    if (Name==NULL || Value==NULL) return NULL;

    if (List==NULL) {    /* if no record in the list */
        List=(Record*)malloc(sizeof(Record));
        if (List!=NULL) {    /* allocate successfully */
            /* next record is NULL */
            List->Next=NULL;

            /* Copy the Name */
            List->Name=(char*)malloc(strlen(Name)+1);
            if (List->Name==NULL) { /* if fail */
                free(List);
                return NULL;
            }
            strcpy(List->Name,Name);

            /* Copy the Value */
            List->Value=(char*)malloc(strlen(Value)+1);
            if (List->Value==NULL) {    /* if fail */
                DeleteRecord(List);
                return NULL;
            }
            strcpy(List->Value,Value);
        }
    }
    else if (!strcmp(List->Name,Name)) {
    /* if the Names are the same, update the record by replacing the value*/

        free(List->Value);

        /*  copy the Value */
        List->Value=(char*)malloc(strlen(Value)+1);
        if (List->Value==NULL) { /* if fail */
            List->Value=NULL;
            DeleteRecord(List);
            return NULL;
        }
        strcpy(List->Value,Value);
    }
    else { /* Otherwise, add the field to the end of the list */
        List->Next=PutRecord(List->Next, Name, Value);
        if (List->Next==NULL) {
            DeleteRecord(List);
            return NULL;
        }
    }
    return List;
}

/************************************************

    Function Trim removes the leading and tailing spaces of a string.
    Parameters:
        s: the string to trim.

************************************************/
void Trim(char *s)
{
    int i,j;

    /*  removing the tailing spaces */
    i=strlen(s)-1;
    while (i>=0 && (s[i]==' ' || s[i]=='\t' || s[i]=='\r')) i--;
    s[i+1]=0;

    /*removing the heading spaces */
    i=0;
    while (s[i]!=0 && (s[i]==' ' || s[i]=='\t' || s[i]=='\r')) i++;
    j=0;
    while (s[i]!=0) s[j++]=s[i++];
    s[j]=0;
}

/*************************************************

    Function Parse parse the parameter file and get the parameter list.
    Returns the parameter list.
    Parameters:
        FileName: the file to parse;
        Separator: the separator between the Name and Value

*************************************************/
Record *Parse(char *FileName, char Separator)
{
    Record *List=NULL;
    FILE *f=fopen(FileName,"r");
    char *Name, *Value, c=' ';
    static char Buffer[10000];
    int Line=0, i, isvalue;

    if (f==NULL) {
        fprintf(stderr,"File %s open error!\n",FileName);
        return NULL;
    }

    /*  parse every line */
    while (c!=EOF) {
        /* Read in one line */
        i=isvalue=0;
        while ((c=getc(f))!='\n' && c!=EOF) {
            if (c==Separator) isvalue=1;
            Buffer[i++]=(isvalue)?c:toupper(c);
        }
        Buffer[i]=0;

        Trim(Buffer);
        Line++;

        /* if the line is empty or starts with "//", it is a comment */
        if (Buffer[0]=='/' && Buffer[1]=='/' || Buffer[0]==0)
            continue;

        /*   Search for Name and Value */
        i=0;
        while (Buffer[i]!=Separator && Buffer[i]!=0) i++;
        if (Buffer[i]==0) {  /* No separator at all, ignore this line */
            fprintf(stderr, "No parameter defined in line %d.\n",Line);
            continue;
        }

        /*Get the name and value */
        Buffer[i]=0;
        Name=Buffer;
        Value=&Buffer[i+1];
        Trim(Name);
        Trim(Value);

        List=PutRecord(List, Name, Value);
        if (List==NULL) {
            fprintf(stderr, "Error at parameter %s, line %d\n",Name,Line);
            break;
        }
    }
    fclose(f);
    return List;
}

/*************************************************

    Function GetInt get the int value of a parameter.
    Returns 0 if fail to find the parameter, 1 if succeed.
    Parameters:
        List: the list to search;
        Name: the name of the parameter;
        Res: the pointer a int to store the value.

*************************************************/
int GetInt(Record *List, char *Name, int *Res) {
    char *Value=SearchRecord(List, Name);
    if (Value==NULL) {
        return 0;
    }
    return sscanf(Value,"%d",Res)!=0;
}

/*************************************************

    Function GetFloat get the float value of a parameter.
    Returns 0 if fail to find the parameter, 1 if succeed.
    Parameters:
        List: the list to search;
        Name: the name of the parameter;
        Res: the pointer a float to store the value.

*************************************************/
int GetFloat(Record *List, char *Name, float *Res) {
    char *Value=SearchRecord(List, Name);
    if (Value==NULL) {
        return 0;
    }
    return sscanf(Value,"%f",Res)!=0;
}

int GetDouble(Record *List, char *Name, double *Res) {
    char *Value=SearchRecord(List, Name);
    if (Value==NULL) {
        return 0;
    }
    return sscanf(Value,"%lf",Res)!=0;
}

/*************************************************

    Function GetBOOL get the BOOL value of a parameter.
    Returns 0 if fail to find the parameter, 1 if succeed.
    Parameters:
        List: the list to search;
        Name: the name of the parameter;
        Res: the pointer a BOOL to store the value.

*************************************************/
int GetBOOL(Record *List, char *Name, int *Res) {
    char *Value=SearchRecord(List, Name);
    if (Value==NULL) {
        return 0;
    }
    if (toupper(Value[0])=='T' || toupper(Value[0])=='F')
        *Res=toupper(Value[0])=='T';
    else
        return 0;
    return 1;
}

/*************************************************

    Function GetString get the string value of a parameter.
    Returns the value, or NULL if fail to find the parameter.
    Parameters:
        List: the list to search;
        Name: the name of the parameter;

*************************************************/
char *GetString(Record *List, char *Name) {
    return SearchRecord(List, Name);
}

#ifdef Test
void main(int args, char *argv[]) {
    int i;
    Record *List;
    int ir;
    float fr;
    int br;
    char *sr;

    if (args>2) {
        List=Parse(argv[1],argv[2][0]);
        if (List==NULL)
            printf("No parameters defined\n");
        else
        for (i=3;i<args;i++) {
            if (GetInt(List,argv[i],&ir))
                printf("Parameter %s (int) = %d\n",argv[i],ir);
         else
            printf("Field %s not found\n",argv[i]);
        if (GetFloat(List,argv[i],&fr))
            printf("Parameter %s (float) = %f\n",argv[i],fr);
         else
            printf("Field %s not found\n",argv[i]);
        if (GetBOOL(List,argv[i],&br))
            printf("Parameter %s (BOOL) = %d\n",argv[i],br);
         else
            printf("Field %s not found\n",argv[i]);
        if ((sr=GetString(List,argv[i]))!=NULL)
            printf("Parameter %s (String) = %s\n",argv[i],sr);
         else
            printf("Field %s not found\n",argv[i]);
      }
    }
   else
    printf("Usage: %s <File> <separater> [parameter1 [Parameter2 [...]]]\n",argv[0]);

   DeleteRecord(List);
}
#endif

