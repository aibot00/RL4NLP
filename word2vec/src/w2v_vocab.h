/***************************************************************************
 * 
 * Copyright (c) 2018aibot.me, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file w2v_vocab.h
 * @author wangyan(darrenwang00@163.com)
 * @date 2018/04/02 11:51:17
 * @version $Revision$ 
 * @brief vocab dict
 *  
 **/

#ifndef _ML_W2V_W2V_VOCAB_H_
#define _ML_W2V_W2V_VOCAB_H_

#include "w2v_define.h"
#include "w2v_option.h"


namespace ml
{


//输出层 lookup table
class VocabHash
{
public:
    VocabHash();
    virtual ~VocabHash();

    int init(W2VOption* opt);

    int destroy();

    //hash word
    int get_word_hash(char* word)
    {
        unsigned long long a, hash = 0;
        for (a = 0; a < strlen(word); a++) {
            hash = hash * 257 + word[a];
        }
        hash = hash % vocab_hash_size;
        return hash; 
    }

    //search
    //Returns position of a word in the vocabulary
    //if the word is not found, returns -1
    int search_vocab(char* word)
    {
        unsigned int hash = get_word_hash(word);
        while (1) {
            if (vocab_hash[hash] == -1) {
                return -1;
            } 
            if (!strcmp(word, vocab[vocab_hash[hash]].word)) {
                return vocab_hash[hash];
            }
            hash = (hash + 1) % vocab_hash_size;
        }
        return -1;
    }

    int add_word2vocab(char * word);

    //frequency using word counts
    int sort_vocab();
    
    //Reduces the vocabulary by removing infrequent tokens
    int reduce_vocab();

    int save_vocab(char* file_name);

    int load_vocab(const char* file_name);

    int learn_vocab_from_train(const char* file_name);

public:
    int *vocab_hash = NULL;
    struct vocab_word *vocab = NULL;
    long long vocab_size = 0;
    long long vocab_max_size = 1000;

    W2VOption* opt = NULL;
};


};

#endif  // _ML_W2V_W2V_VOCAB_H_

/* vim: set ts=4 sw=4 sts=4 tw=100 */

