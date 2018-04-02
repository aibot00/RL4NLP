/***************************************************************************
 * 
 * Copyright (c) 2018aibot.me, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file w2v_vocab.cpp
 * @author wangyan(darrenwang00@163.com)
 * @date 2018/04/02 11:51:17
 * @version $Revision$ 
 * @brief vocab dict
 *  
 **/
#include "w2v_vocab.h"
#include "w2v_util.h"


namespace ml
{


VocabHash::VocabHash(): vocab_hash(NULL), vocab(NULL), vocab_size(0), vocab_max_size(1000), opt(NULL)
{
}

VocabHash::~VocabHash()
{
    this->destroy();
}


int VocabHash::init(W2VOption* opt)
{
    if (opt == NULL) {
        return -1;
    }
    this->opt = opt;

    //init vocab hash dict
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

    return RET_SUCC;
} 


int VocabHash::destroy()
{
    //clean vocab node buf
    for (int i = 0; i < vocab_size; ++i)
    {
        if (vocab[i].word != NULL) {
            free(vocab[i].word);
        }
        if (vocab[i].point != NULL) {
            free(vocab[i].point);
        }
        if (vocab[i].code != NULL) {
            free(vocab[i].code);
        }
    }

    if (vocab != NULL) {
        free(vocab);
        vocab = NULL;
    }
    if (vocab_hash != NULL) {
        free(vocab_hash);
        vocab_hash = NULL;
    }
    vocab_size = 0;

    return RET_SUCC;
}



int VocabHash::add_word2vocab(char * word)
{
    unsigned int hash = 0;
    unsigned int length = strlen(word) + 1;

    if (length > MAX_STRING) length = MAX_STRING;

    //copy word & index++
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;

    //reallocate memory
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        size_t max_byte = vocab_max_size * sizeof(struct vocab_word);
        vocab = (struct vocab_word *)realloc(vocab, max_byte);
        if (vocab == NULL) {
            std::cerr << "realloc memory error! with size=" << max_byte << std::endl;
            return -1;
        }
    }
    
    //add 
    hash = get_word_hash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}


int VocabHash::sort_vocab()
{
    int i = 0;
    int size = 0;
    unsigned int hash = 0;

    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), vocab_cmp);
    for (i = 0; i < vocab_hash_size; ++i)  {
        vocab_hash[i] = -1;
    }

    size = vocab_size;
    opt->train_words = 0;

    for (i = 0; i < size; ++i)
    {
        // Words occuring less than min_count times will be discarded from the vocab
        if ((vocab[i].cn < opt->min_count) && (i != 0))
        {
            vocab_size--;
            free(vocab[i].word);
        }
        else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash = get_word_hash(vocab[i].word);
            while (vocab_hash[hash] != -1) {
                hash = (hash + 1) % vocab_hash_size;
            }
            vocab_hash[hash] = i;
            opt->train_words += vocab[i].cn;
        }
    }
    
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    if (vocab == NULL) {
        std::cerr << "realloc sort_vocab dict error" << std::endl;
        return RET_FAIL;
    }

    //allocate memory for the binary tree
    for (i = 0; i < vocab_size; ++i) {
        vocab[i].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[i].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }

    return RET_SUCC;
}


// Reduces the vocabulary by removing infrequent tokens
int VocabHash::reduce_vocab()
{
    if (vocab == NULL || vocab_hash == NULL || opt == NULL) {
        return RET_FAIL;
    }
    int i =0;
    int j = 0;
    unsigned int hash = 0;

    //rm low freq
    for (i = 0; i < vocab_size; ++i)
    {     
        if (vocab[i].cn > opt->min_reduce) {
            vocab[j].cn = vocab[i].cn;
            vocab[j].word = vocab[i].word;
            j++;
        } else {
            free(vocab[i].word);
        }
    }

    vocab_size = j;
    for (i = 0; i < vocab_hash_size; i++) vocab_hash[i] = -1;

    for (i = 0; i < vocab_size; ++i)
    {
        // Hash will be re-computed, as it is not actual
        hash = get_word_hash(vocab[i].word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = i;
    }
    fflush(stdout);
    opt->min_reduce++;
    return RET_SUCC;
}


int VocabHash::save_vocab(char* file_name)
{
    if (file_name == NULL) {
        return RET_FAIL;
    }
    long long i;
    FILE *fo = fopen(file_name, "wb");
    for (i = 0; i < vocab_size; i++) {
        fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    }
    fclose(fo);
    return RET_SUCC;
}


int VocabHash::load_vocab(const char* file_name)
{
    if (file_name == NULL) {
        return RET_FAIL;
    }

    char c;
    long long a, i = 0;
    char word[MAX_STRING] = {0};
    
    FILE *fin = fopen(file_name, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    
    vocab_size = 0;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    while (1)
    {
        read_word(word, fin);
        if (feof(fin)) break;
        a = add_word2vocab(word);
        if (a < 0) {
            std::cerr << "load_vocab->add_word2vocab error!" << std::endl;
            fclose(fin);
            return RET_FAIL;
        }
        fscanf(fin, "%lld%c", &vocab[a].cn, &c);
        i++;
    }
    fclose(fin);

    this->sort_vocab();
    if (opt->debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", opt->train_words);
    }

    opt->file_size = file_size(opt->train_file);
    return RET_SUCC;
}


int VocabHash::learn_vocab_from_train(const char* file_name)
{
    if (file_name == NULL) {
        return RET_FAIL;
    }
    long long a, i;
    char word[MAX_STRING] = {0};
    FILE *fin = NULL;

    for (i = 0; i < vocab_hash_size; ++i) vocab_hash[i] = -1;

    fin = fopen(opt->train_file, "rb");
    if (fin == NULL) {
        std::cerr << "ERROR: training data file not found!" << std::endl;
        return RET_FAIL;
    }

    vocab_size = 0;
    add_word2vocab((char *)"</s>");
    while (1)
    {
        read_word(word, fin);
        if (feof(fin)) break;

        opt->train_words++;
        if ((opt->debug_mode > 1) && (opt->train_words % 100000 == 0))
        {
            printf("%lldK%c", opt->train_words / 1000, 13);
            fflush(stdout);
        }
        i = search_vocab(word);
        if (i == -1) {
            a = add_word2vocab(word);
            vocab[a].cn = 1;
        } else {
            vocab[i].cn++;
        }
        if (vocab_size > vocab_hash_size * 0.7) {
            reduce_vocab();
        }
    }

    sort_vocab();
    if (opt->debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", opt->train_words);
    }

    opt->file_size = ftell(fin);
    fclose(fin);
    return RET_SUCC;
}


};

/* vim: set ts=4 sw=4 sts=4 tw=100 */