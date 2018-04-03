/***************************************************************************
 * 
 * Copyright (c) 2018aibot.me, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file w2v_define.h
 * @author wangyan(darrenwang00@163.com)
 * @date 2018/04/02 11:51:16
 * @version $Revision$ 
 * @brief common define
 *  
 **/

#ifndef _ML_W2V_W2V_DEFINE_H_
#define _ML_W2V_W2V_DEFINE_H_

#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#include <time.h>


namespace ml
{

////////////// define ///////////////

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

#define RET_SUCC 0
#define RET_FAIL -1

typedef float real;                    // Precision of float number
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabular


//word2vec word struct
struct vocab_word
{
    vocab_word(): word(NULL), code(NULL), codelen(0), point(NULL), cn(0) { }
    char *word;   
    char *code;   //humffman code
    char codelen; //huffman code len
    int  *point;  //for root to level
    long long cn; //word cnt
};


};

#endif  // _ML_W2V_W2V_DEFINE_H_

/* vim: set ts=4 sw=4 sts=4 tw=100 */

