/***************************************************************************
 * 
 * Copyright (c) 2018aibot.me, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file w2v_model.h
 * @author wangyan(darrenwang00@163.com)
 * @date 2018/04/02 11:51:17
 * @version $Revision$ 
 * @brief word2vec model
 *  
 **/


#ifndef _ML_W2V_W2V_MODEL_H_
#define _ML_W2V_W2V_MODEL_H_

#include "w2v_define.h"
#include "w2v_option.h"
#include "w2v_vocab.h"
#include "w2v_util.h"


namespace ml
{


class W2VModel
{
public:
    W2VModel();
    virtual ~W2VModel();
    
    static W2VModel* get_instance();

    int parser(int argc, char* argv[]);

    int train();

    int save();

    int destroy();

    //train model thread
    int run_thread(void* id);
    
private:
    W2VModel(const W2VModel&);
    W2VModel& operator=(const W2VModel&);

    int init();
    int init_net();
    int create_huffman_tree();
    int init_unigram_table();

    //read a word and return index int dict
    int read_word_index(FILE* fin)
    {
        char word[MAX_STRING];
        read_word(word, fin);
        if (feof(fin)) {
            return -1;
        }
        return vocab_dic.search_vocab(word);
    }
    
    static W2VModel *_instance;

public:
    bool is_init;

    W2VOption  opt;

    //data
    VocabHash  vocab_dic;
    
    long long word_count_actual;
    
    //matrix
    real *syn0;
    real *syn1;
    real *syn1neg;
    real *expTable;   //sigmoid table
    
    const int table_size = 1e8;
    int *table;
    clock_t start;
};


};

#endif  // _ML_W2V_W2V_MODEL_H_

/* vim: set ts=4 sw=4 sts=4 tw=100 */

