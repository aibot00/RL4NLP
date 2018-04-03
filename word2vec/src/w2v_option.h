/***************************************************************************
 * 
 * Copyright (c) 2018aibot.me, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file w2v_option.h
 * @author wangyan(darrenwang00@163.com)
 * @date 2018/04/02 11:51:16
 * @version $Revision$ 
 * @brief option 
 *  
 **/

#ifndef _ML_W2V_W2V_OPTION_H_
#define _ML_W2V_W2V_OPTION_H_

#include "w2v_define.h"


namespace ml
{


// option
class W2VOption
{
public:
    W2VOption();
    virtual ~W2VOption();

    int reset();
    
    static int help_info();
    
    int parser_cmd(int argc, char **argv);
    
    std::string debug();
private:
    int parse_one(char *str, int argc, char **argv); 

public:
    //param
    char train_file[MAX_STRING];      //train data
    char output_file[MAX_STRING];     //model file
    char save_vocab_file[MAX_STRING]; //save word dict
    char read_vocab_file[MAX_STRING]; //user word dict

    int hs;      //use Hierarchical Softmax 
    int cbow;    //continuous bag of words model 
    int window;  //max skip length between words

    int negative;
    long long iter;
    long long layer1_size;

    real alpha;
    real sample;
    real starting_alpha;

    int binary;
    int debug_mode;
    int min_count;
    int num_threads;
    int min_reduce;
    long long classes;

    //data
    long long train_words;
    long long file_size;
};


}; // end of namespace ml

#endif  // _ML_W2V_W2V_OPTION_H_

/* vim: set ts=4 sw=4 sts=4 tw=100 */
