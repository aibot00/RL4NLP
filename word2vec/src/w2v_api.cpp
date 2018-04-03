/***************************************************************************
 * 
 * Copyright (c) 2018aibot.me, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file w2v_api.cpp
 * @author wangyan(darrenwang00@163.com)
 * @date 2018/04/03 11:51:17
 * @version $Revision$ 
 * @brief word2vec api
 *  
 **/

#include "w2v_api.h"
#include "w2v_define.h"
#include "w2v_model.h"
#include <unordered_map>


namespace ml
{


///////////////////// LookUpTable ///////////////////

class LookUpTable
{
public:
    LookUpTable() : word_num(0), vec_size(0), M(0) { word_idx.clear(); }
    ~LookUpTable() { destroy(); }

    int load(const char* file_name);

    int destroy() {
        word_num = 0;
        vec_size = 0;
        if (M != NULL) {
            free(M); M = NULL;
        }
        word_idx.clear();
    }

public:
    long long word_num;
    long long vec_size;
    float *M;
    std::unordered_map<std::string, uint32_t>  word_idx;
};


int LookUpTable::load(const char* model_file)
{
    if (model_file == NULL) {
        return -1;
    }
    FILE* infile = fopen(model_file, "rb");
    if (infile == NULL) {
        std::cerr << "ERROR: w2v_load param error!" << std::endl;
        return -1;
    }

    long long j = 0;
    word_num = 0, vec_size = 0;

    fscanf(infile, "%lld", &word_num);
    fscanf(infile, "%lld", &vec_size);

    size_t matrix_size = (long long)word_num * (long long)vec_size * sizeof(float);
    M = (float *)malloc(matrix_size);
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",  matrix_size / 1048576, word_num, vec_size);
        return -1;
    }

    char word[64] = {0};
    uint32_t offset = 0;
    float len = 0.0;

    for (long long i = 0; i < word_num; ++i)
    {
        j = 0;
        while (1) {
            word[j] = fgetc(infile);
            if (feof(infile) || (word[j] == ' ')) break;
            if ((j < 64) && (word[j] != '\n')) j++;
        }
        word[j] = 0;
        offset = i * vec_size;

        fread(&M[offset], sizeof(float), vec_size, infile);

        len = 0;
        for (j = 0; j < vec_size; j++) len += M[offset + j] * M[offset + j];
        len = sqrt(len);
        for (j = 0; j < vec_size; j++) M[offset + j] /= len;

        //add to hash
        this->word_idx[word] = offset;
    }
    
    fclose(infile);
    return 0;
}


///////////////////// api ///////////////////

int w2v_train(int argc, char* argv[])
{
    if (argc < 2 || argv == NULL) {
        std::cerr << "ERROR: w2v_train param error!" << std::endl;
        W2VOption::help_info(); 
        return -1;
    }

    W2VModel* m = W2VModel::get_instance();
    if (m == NULL || m->parser(argc, argv) != RET_SUCC)
    {
        std::cerr << "ERROR: w2v_train parser param error!" << std::endl;
        return -2;
    }

    if (m->train() != RET_SUCC)
    {
        std::cerr << "ERROR: w2v_train train model error!" << std::endl;
        return -3;
    }
    
    //clean
    delete m;
    m = NULL;

    std::cout << "tain model done" << std::endl;
    return 0;
}



void* w2v_load(const char* model_file)
{
    LookUpTable* model = new LookUpTable();
    if (model == NULL || model->load(model_file) != 0) {
        return NULL;
    }
    return (void*)model;
}

int w2v_destroy(void* model)
{
    if (model == NULL) {
        return -1;
    }
    delete (LookUpTable*)model;
    model = NULL;
    return 0;
}


int w2v_search(void* model, const char* word, float* &vec, int & size)
{
    if (model == NULL || word == NULL) {
        return -1;
    }
    LookUpTable* table = (LookUpTable*)model;

    vec = NULL;
    size = 0;
    if (table->word_idx.find(word) == table->word_idx.end()) {
        return -1;
    }

    uint32_t offset = table->word_idx[word];
    vec = &(table->M[offset]);
    size = table->vec_size;
    return 0;
}


};

/* vim: set ts=4 sw=4 sts=4 tw=100 */

