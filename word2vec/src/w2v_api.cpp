/***************************************************************************
 * 
 * Copyright (c) 2018aibot.me, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file w2v_api.cpp
 * @author wangyan(darrenwang00@163.com)
 * @date 2018/04/02 11:51:17
 * @version $Revision$ 
 * @brief word2vec api
 *  
 **/

#include "w2v_api.h"
#include "w2v_define.h"
#include "w2v_model.h"


namespace ml
{


int w2v_train(int argc, char* argv[])
{
    if (argc < 2 || argv == NULL) {
        std::cerr << "ERROR: w2v_train param error!" << std::endl;
        return -1;
    }

    W2VModel* m = W2VModel::get_instance();
    if (m == NULL ||  m->parser(argc, argv) != RET_SUCC)
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
    if (model_file == NULL) {
        return NULL;
    }
    return NULL;
}


int w2v_search(void* model, const char* word, float* &vec, int & size)
{
    if (model == NULL || word == NULL) {
        return -1;
    }
    vec = NULL;
    size = 0;

    return 0;
}


int w2v_destroy(void* model)
{
    if (model == NULL) {
        return -1;
    }

    return 0;
}


};

/* vim: set ts=4 sw=4 sts=4 tw=100 */

