/***************************************************************************
 * 
 * Copyright (c) 2018aibot.me, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file w2v_api.h
 * @author wangyan(darrenwang00@163.com)
 * @date 2018/04/02 11:51:17
 * @version $Revision$ 
 * @brief word2vec api
 *  
 **/

namespace ml
{

int w2v_train(int argc, char* argv[]);

void* w2v_load(const char* model_file);

int w2v_search(void* model, const char* word, float* &vec, int & size);

int w2v_destroy(void* model);


};

/* vim: set ts=4 sw=4 sts=4 tw=100 */