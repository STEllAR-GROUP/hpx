#include <sys/cdefs.h>
#ifndef lint
#if 0
static char yysccsid[] = "@(#)yaccpar	1.9 (Berkeley) 02/21/93";
#else
__IDSTRING(yyrcsid, "$NetBSD: skeleton.c,v 1.14 1997/10/20 03:41:16 lukem Exp $");
#endif
#endif
#include <stdlib.h>
#define YYBYACC 1
#define YYMAJOR 1
#define YYMINOR 9
#define YYLEX yylex()
#define YYEMPTY -1
#define yyclearin (yychar=(YYEMPTY))
#define yyerrok (yyerrflag=0)
#define YYRECOVERING (yyerrflag!=0)
#define YYPREFIX "yy"
#define N3_QNAME 257
#define N3_EXPLICITURI 258
#define N3_VARIABLE 259
#define N3_NUMERICLITERAL 260
#define N3_STRING 261
#define N3_BARENAME 262
#define YYERRCODE 256
short yylhs[] = {                                        -1,
    0,    1,    1,    2,    2,    3,    3,    4,    4,    6,
    6,    7,    8,    9,   10,   10,   11,   11,   12,   12,
   13,   14,   14,   15,   15,   16,   16,   17,   17,   18,
   18,   19,   20,   20,   20,   20,   20,   20,   20,   21,
   22,   22,   23,   23,   24,   24,   24,   24,   24,   24,
   24,   24,   25,   25,   25,   26,   27,   28,   29,   29,
   30,   30,   31,   31,   32,   32,   33,   33,   34,   34,
   34,   35,   35,    5,
};
short yylen[] = {                                         2,
    5,    0,    2,    0,    2,    0,    2,    0,    3,    4,
    2,    2,    2,    2,    1,    2,    1,    2,    1,    2,
    1,    0,    4,    1,    3,    1,    1,    1,    3,    1,
    3,    2,    1,    2,    3,    1,    1,    1,    1,    1,
    0,    3,    0,    5,    1,    3,    1,    1,    1,    3,
    3,    1,    0,    2,    2,    1,    4,    2,    0,    2,
    0,    2,    0,    2,    0,    2,    0,    2,    0,    2,
    2,    0,    2,    0,
};
short yydefred[] = {                                      0,
    0,    0,    0,    0,    0,    0,    0,   15,   11,    0,
    0,    0,    3,    0,   24,    0,   16,   27,   26,   17,
   12,    0,    0,    0,    0,    5,   10,    0,   28,    0,
   18,   19,   13,    0,   47,   48,    0,    0,    0,    0,
   52,   74,    0,    0,   45,   21,    0,   49,    7,   25,
    0,   30,    0,   20,    0,    0,   58,    0,    0,    0,
    0,    0,   36,   37,   38,   39,    0,    0,   56,   33,
    0,    0,    1,    0,   14,    0,    0,   32,   29,    0,
   70,   71,   62,   46,    0,    0,   34,    0,   50,   40,
    0,   60,   51,    9,   54,   55,   31,   64,    0,    0,
   35,    0,    0,   66,    0,   57,    0,    0,   23,    0,
   68,   42,    0,   73,    0,    0,   44,
};
short yydgoto[] = {                                       3,
    4,   11,   24,   42,   73,    5,   12,   25,   43,    9,
   21,   33,   44,   67,   17,   45,   31,   54,   46,   68,
   91,  103,  109,   47,   78,   70,   59,   48,   72,   60,
   86,  100,  106,   57,  111,
};
short yysindex[] = {                                   -206,
 -237,  -46,    0, -242, -206, -232,   -4,    0,    0,  -44,
 -227, -242,    0,   -3,    0, -233,    0,    0,    0,    0,
    0,    4,  -40,   74, -227,    0,    0,   -4,    0, -198,
    0,    0,    0,    9,    0,    0,  -55, -206,   45,   74,
    0,    0,    6,   45,    0,    0,  -23,    0,    0,    0,
    4,    0, -198,    0,   -2, -198,    0, -206,  -58, -242,
   74,   74,    0,    0,    0,    0,  -16,   74,    0,    0,
   74,   29,    0,   74,    0,   74,   74,    0,    0,    9,
    0,    0,    0,    0, -242, -227,    0, -191,    0,    0,
   36,    0,    0,    0,    0,    0,    0,    0, -227,   74,
    0,   74,   25,    0,   44,    0,   36,   45,    0,   74,
    0,    0,   74,    0,   36,   25,    0,
};
short yyrindex[] = {                                      1,
    0,    0,    0,   11,    1,    0,    0,    0,    0,    0,
   21,   11,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,   93,   21,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,  -28,  -10,    5,   58,
    0,    0,    0,  -42,    0,    0,   28,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,  -10,    0,   63,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   58,    0,    0,   93,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,   63,  -37,    0,    0,    0,    0,
  -31,    0,    0,    0,    0,    0,    0,    0,  -37,  -25,
    0,    0,  -29,    0,  -20,    0,  -31,    0,    0,  -25,
    0,    0,    0,    0,  -31,  -29,    0,
};
short yygindex[] = {                                      0,
  102,   96,   86,   42,    0,  -24,  -53,  -62,  -75,    0,
    0,    0,    0,   73,   90,   26,   69,   43,   33,   17,
  -94,  -69,   10,  -17,    0,   14,    0,    0,   51,   70,
   46,   30,   20,    0,    0,
};
#define YYTABLESIZE 347
short yytable[] = {                                       8,
    2,   20,   65,   22,   69,   32,   85,  107,   55,   76,
    4,   69,   69,   58,   41,   69,   43,   69,  115,    6,
    6,   69,   10,   99,  105,   14,   69,   41,   28,   61,
   69,   85,   69,   58,  105,   22,   99,  112,   23,   16,
    2,   15,   27,   69,   69,  116,   81,   30,   34,   29,
    4,   74,   53,   65,   52,   51,    1,    2,   18,   19,
    6,   41,   69,   43,   69,   69,   84,   53,   53,   93,
   77,   53,   71,   53,   87,   88,   89,  101,   80,  102,
   61,   82,   22,  108,   40,   65,   53,   65,   53,  110,
   69,    2,    8,   41,   69,   43,   69,   22,   59,   67,
   90,    4,   63,   71,   72,   64,   13,   26,   95,   96,
   49,    6,   61,   40,   61,   94,   75,   50,   53,   79,
   53,   92,   97,    2,  113,  117,    0,   83,  104,  114,
   98,    0,    0,    4,   90,   39,    0,    0,    0,    0,
    0,    0,    0,    6,    0,   90,    0,    0,    0,    0,
   53,    0,   53,   63,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,   39,    0,    0,   38,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,   63,    0,   63,    0,    0,
    0,    0,    0,    0,    0,    0,   38,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,   18,   19,    0,    7,   18,   19,   56,   65,
   65,   65,   65,   65,    0,    0,    0,    0,   69,   69,
   69,   69,   69,    0,    0,   65,    0,    0,   69,   69,
   69,   69,   69,   69,   69,    0,   61,   61,   61,   61,
   61,    0,    0,    0,   61,   61,    0,    2,    2,    2,
    2,    2,   61,    0,    0,    2,    2,    4,    4,    4,
    4,    4,    0,    2,    0,    0,    4,    6,    6,    6,
    6,    6,    0,    4,   53,   53,   53,   53,   53,    0,
    0,    0,    0,    6,   53,   53,    0,   53,   53,   53,
   53,   18,   19,   35,   36,   37,    0,    0,    0,    0,
    0,   61,   62,    0,   63,   65,   66,   41,    0,   63,
   63,   63,   63,   63,    0,    0,    0,    0,   63,    0,
   18,   19,   35,   36,   37,   63,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   41,
};
short yycheck[] = {                                      46,
    0,   46,   40,   46,   33,   46,   60,  102,   64,   33,
    0,   40,   41,   38,   46,   44,   46,   46,  113,  257,
    0,   39,  265,   86,  100,  258,   44,   59,  262,   40,
   59,   85,   61,   58,  110,   10,   99,  107,  266,   44,
   40,   46,   46,   61,   62,  115,   49,   44,   23,   46,
   40,   46,   44,   91,   46,   30,  263,  264,  257,  258,
   40,   93,   91,   93,   93,   94,  125,   40,   41,   41,
   94,   44,   40,   46,   61,   62,   93,  269,   53,   44,
   91,   56,  125,   59,   40,  123,   59,  125,   61,   46,
  108,   91,    0,  125,  123,  125,  125,   93,   41,  125,
   68,   91,   40,   71,  125,   61,    5,   12,   76,   77,
   25,   91,  123,   40,  125,   74,   44,   28,   91,   51,
   93,   71,   80,  123,  108,  116,   -1,   58,   99,  110,
   85,   -1,   -1,  123,  102,   91,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,  123,   -1,  113,   -1,   -1,   -1,   -1,
  123,   -1,  125,   91,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   91,   -1,   -1,  123,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,  123,   -1,  125,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,  123,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,  257,  258,   -1,  262,  257,  258,  274,  257,
  258,  259,  260,  261,   -1,   -1,   -1,   -1,  257,  258,
  259,  260,  261,   -1,   -1,  273,   -1,   -1,  267,  268,
  269,  270,  271,  272,  273,   -1,  257,  258,  259,  260,
  261,   -1,   -1,   -1,  265,  266,   -1,  257,  258,  259,
  260,  261,  273,   -1,   -1,  265,  266,  257,  258,  259,
  260,  261,   -1,  273,   -1,   -1,  266,  257,  258,  259,
  260,  261,   -1,  273,  257,  258,  259,  260,  261,   -1,
   -1,   -1,   -1,  273,  267,  268,   -1,  270,  271,  272,
  273,  257,  258,  259,  260,  261,   -1,   -1,   -1,   -1,
   -1,  267,  268,   -1,  270,  271,  272,  273,   -1,  257,
  258,  259,  260,  261,   -1,   -1,   -1,   -1,  266,   -1,
  257,  258,  259,  260,  261,  273,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,  273,
};
#define YYFINAL 3
#ifndef YYDEBUG
#define YYDEBUG 0
#endif
#define YYMAXTOKEN 274
#if YYDEBUG
char *yyname[] = {
"end-of-file",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
"'!'",0,0,0,0,0,0,"'('","')'",0,0,"','",0,"'.'",0,0,"'1'",0,0,0,0,0,0,0,0,0,
"';'",0,"'='",0,0,"'@'",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
"'['",0,"']'","'^'",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
"'{'",0,"'}'",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"N3_QNAME","N3_EXPLICITURI","N3_VARIABLE",
"N3_NUMERICLITERAL","N3_STRING","N3_BARENAME","\"@prefix\"","\"@keywords\"",
"\"@forAll\"","\"@forSome\"","\"@has\"","\"@is\"","\"@of\"","\"@a\"","\"=>\"",
"\"<=\"","\"@this\"","\"^^\"",
};
char *yyrule[] = {
"$accept : n3_document",
"n3_document : _g0 _g1 _g2 n3_statements_optional eof",
"_g0 :",
"_g0 : n3_declaration _g0",
"_g1 :",
"_g1 : n3_universal _g1",
"_g2 :",
"_g2 : n3_existential _g2",
"n3_statements_optional :",
"n3_statements_optional : n3_statement '.' n3_statements_optional",
"n3_declaration : \"@prefix\" N3_QNAME N3_EXPLICITURI '.'",
"n3_declaration : \"@keywords\" _g8",
"n3_universal : \"@forAll\" _g6",
"n3_existential : \"@forSome\" _g7",
"n3_statement : n3_subject n3_propertylist",
"_g8 : '.'",
"_g8 : N3_BARENAME _g11",
"_g6 : '.'",
"_g6 : n3_symbol _g9",
"_g7 : '.'",
"_g7 : n3_symbol _g10",
"n3_subject : n3_path",
"n3_propertylist :",
"n3_propertylist : n3_verb n3_object n3_objecttail n3_propertylisttail",
"_g11 : '.'",
"_g11 : ',' N3_BARENAME _g11",
"n3_symbol : N3_EXPLICITURI",
"n3_symbol : N3_QNAME",
"_g9 : '.'",
"_g9 : ',' n3_symbol _g9",
"_g10 : '.'",
"_g10 : ',' n3_symbol _g10",
"n3_path : n3_node n3_pathtail",
"n3_verb : n3_prop",
"n3_verb : \"@has\" n3_prop",
"n3_verb : \"@is\" n3_prop \"@of\"",
"n3_verb : \"@a\"",
"n3_verb : '='",
"n3_verb : \"=>\"",
"n3_verb : \"<=\"",
"n3_object : n3_path",
"n3_objecttail :",
"n3_objecttail : ',' n3_object n3_objecttail",
"n3_propertylisttail :",
"n3_propertylisttail : ';' n3_verb n3_object n3_objecttail n3_propertylisttail",
"n3_node : n3_symbol",
"n3_node : '{' n3_formulacontent '}'",
"n3_node : N3_VARIABLE",
"n3_node : N3_NUMERICLITERAL",
"n3_node : n3_literal",
"n3_node : '[' n3_propertylist ']'",
"n3_node : '(' n3_pathlist ')'",
"n3_node : \"@this\"",
"n3_pathtail :",
"n3_pathtail : '!' n3_path",
"n3_pathtail : '^' n3_path",
"n3_prop : n3_node",
"n3_formulacontent : _g3 _g4 _g5 n3_statementlist",
"n3_literal : N3_STRING n3_dtlang",
"n3_pathlist :",
"n3_pathlist : n3_path n3_pathlist",
"_g3 :",
"_g3 : n3_declaration _g3",
"_g4 :",
"_g4 : n3_universal _g4",
"_g5 :",
"_g5 : n3_existential _g5",
"n3_statementlist :",
"n3_statementlist : n3_statement n3_statementtail",
"n3_dtlang :",
"n3_dtlang : '@' '1'",
"n3_dtlang : \"^^\" n3_symbol",
"n3_statementtail :",
"n3_statementtail : '.' n3_statementlist",
"eof :",
};
#endif
#ifndef YYSTYPE
typedef int YYSTYPE;
#endif
#ifdef YYSTACKSIZE
#undef YYMAXDEPTH
#define YYMAXDEPTH YYSTACKSIZE
#else
#ifdef YYMAXDEPTH
#define YYSTACKSIZE YYMAXDEPTH
#else
#define YYSTACKSIZE 10000
#define YYMAXDEPTH 10000
#endif
#endif
#define YYINITSTACKSIZE 200
int yydebug;
int yynerrs;
int yyerrflag;
int yychar;
short *yyssp;
YYSTYPE *yyvsp;
YYSTYPE yyval;
YYSTYPE yylval;
short *yyss;
short *yysslim;
YYSTYPE *yyvs;
int yystacksize;
/* allocate initial stack or double stack size, up to YYMAXDEPTH */
int yyparse __P((void));
static int yygrowstack __P((void));
static int yygrowstack()
{
    int newsize, i;
    short *newss;
    YYSTYPE *newvs;

    if ((newsize = yystacksize) == 0)
        newsize = YYINITSTACKSIZE;
    else if (newsize >= YYMAXDEPTH)
        return -1;
    else if ((newsize *= 2) > YYMAXDEPTH)
        newsize = YYMAXDEPTH;
    i = yyssp - yyss;
    if ((newss = (short *)realloc(yyss, newsize * sizeof *newss)) == NULL)
        return -1;
    yyss = newss;
    yyssp = newss + i;
    if ((newvs = (YYSTYPE *)realloc(yyvs, newsize * sizeof *newvs)) == NULL)
        return -1;
    yyvs = newvs;
    yyvsp = newvs + i;
    yystacksize = newsize;
    yysslim = yyss + newsize - 1;
    return 0;
}

#define YYABORT goto yyabort
#define YYREJECT goto yyabort
#define YYACCEPT goto yyaccept
#define YYERROR goto yyerrlab
int
yyparse()
{
    int yym, yyn, yystate;
#if YYDEBUG
    char *yys;

    if ((yys = getenv("YYDEBUG")) != NULL)
    {
        yyn = *yys;
        if (yyn >= '0' && yyn <= '9')
            yydebug = yyn - '0';
    }
#endif

    yynerrs = 0;
    yyerrflag = 0;
    yychar = (-1);

    if (yyss == NULL && yygrowstack()) goto yyoverflow;
    yyssp = yyss;
    yyvsp = yyvs;
    *yyssp = yystate = 0;

yyloop:
    if ((yyn = yydefred[yystate]) != 0) goto yyreduce;
    if (yychar < 0)
    {
        if ((yychar = yylex()) < 0) yychar = 0;
#if YYDEBUG
        if (yydebug)
        {
            yys = 0;
            if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
            if (!yys) yys = "illegal-symbol";
            printf("%sdebug: state %d, reading %d (%s)\n",
                    YYPREFIX, yystate, yychar, yys);
        }
#endif
    }
    if ((yyn = yysindex[yystate]) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yychar)
    {
#if YYDEBUG
        if (yydebug)
            printf("%sdebug: state %d, shifting to state %d\n",
                    YYPREFIX, yystate, yytable[yyn]);
#endif
        if (yyssp >= yysslim && yygrowstack())
        {
            goto yyoverflow;
        }
        *++yyssp = yystate = yytable[yyn];
        *++yyvsp = yylval;
        yychar = (-1);
        if (yyerrflag > 0)  --yyerrflag;
        goto yyloop;
    }
    if ((yyn = yyrindex[yystate]) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yychar)
    {
        yyn = yytable[yyn];
        goto yyreduce;
    }
    if (yyerrflag) goto yyinrecovery;
    goto yynewerror;
yynewerror:
    yyerror("syntax error");
    goto yyerrlab;
yyerrlab:
    ++yynerrs;
yyinrecovery:
    if (yyerrflag < 3)
    {
        yyerrflag = 3;
        for (;;)
        {
            if ((yyn = yysindex[*yyssp]) && (yyn += YYERRCODE) >= 0 &&
                    yyn <= YYTABLESIZE && yycheck[yyn] == YYERRCODE)
            {
#if YYDEBUG
                if (yydebug)
                    printf("%sdebug: state %d, error recovery shifting\
 to state %d\n", YYPREFIX, *yyssp, yytable[yyn]);
#endif
                if (yyssp >= yysslim && yygrowstack())
                {
                    goto yyoverflow;
                }
                *++yyssp = yystate = yytable[yyn];
                *++yyvsp = yylval;
                goto yyloop;
            }
            else
            {
#if YYDEBUG
                if (yydebug)
                    printf("%sdebug: error recovery discarding state %d\n",
                            YYPREFIX, *yyssp);
#endif
                if (yyssp <= yyss) goto yyabort;
                --yyssp;
                --yyvsp;
            }
        }
    }
    else
    {
        if (yychar == 0) goto yyabort;
#if YYDEBUG
        if (yydebug)
        {
            yys = 0;
            if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
            if (!yys) yys = "illegal-symbol";
            printf("%sdebug: state %d, error recovery discards token %d (%s)\n",
                    YYPREFIX, yystate, yychar, yys);
        }
#endif
        yychar = (-1);
        goto yyloop;
    }
yyreduce:
#if YYDEBUG
    if (yydebug)
        printf("%sdebug: state %d, reducing by rule %d (%s)\n",
                YYPREFIX, yystate, yyn, yyrule[yyn]);
#endif
    yym = yylen[yyn];
    yyval = yyvsp[1-yym];
    switch (yyn)
    {
    }
    yyssp -= yym;
    yystate = *yyssp;
    yyvsp -= yym;
    yym = yylhs[yyn];
    if (yystate == 0 && yym == 0)
    {
#if YYDEBUG
        if (yydebug)
            printf("%sdebug: after reduction, shifting from state 0 to\
 state %d\n", YYPREFIX, YYFINAL);
#endif
        yystate = YYFINAL;
        *++yyssp = YYFINAL;
        *++yyvsp = yyval;
        if (yychar < 0)
        {
            if ((yychar = yylex()) < 0) yychar = 0;
#if YYDEBUG
            if (yydebug)
            {
                yys = 0;
                if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
                if (!yys) yys = "illegal-symbol";
                printf("%sdebug: state %d, reading %d (%s)\n",
                        YYPREFIX, YYFINAL, yychar, yys);
            }
#endif
        }
        if (yychar == 0) goto yyaccept;
        goto yyloop;
    }
    if ((yyn = yygindex[yym]) && (yyn += yystate) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yystate)
        yystate = yytable[yyn];
    else
        yystate = yydgoto[yym];
#if YYDEBUG
    if (yydebug)
        printf("%sdebug: after reduction, shifting from state %d \
to state %d\n", YYPREFIX, *yyssp, yystate);
#endif
    if (yyssp >= yysslim && yygrowstack())
    {
        goto yyoverflow;
    }
    *++yyssp = yystate;
    *++yyvsp = yyval;
    goto yyloop;
yyoverflow:
    yyerror("yacc stack overflow");
yyabort:
    return (1);
yyaccept:
    return (0);
}
