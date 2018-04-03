/***************************************************************************
 * 
 * Copyright (c) 2018aibot.me, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file test.cpp
 * @author wangyan(darrenwang00@163.com)
 * @date 2018/04/02 11:51:17
 * @version $Revision$ 
 * @brief  test model
 **/
#include "w2v_api.h"
#include <iostream>


int sample()
{
    void* m = ml::w2v_load("./model2");

    float* vec;
    int size;
    if (ml::w2v_search(m, "</s>", vec, size) != 0) {
        std::cout << "cant not find  key" << std::endl;
    }
    
    for (int i =0; i < size; ++i) {
        std::cout << vec[i] << " ";
        std::cout << " 1111 " << std::endl;
    }
    ml::w2v_destroy(m);
    return 0;
}


///// mian /////
int main(int argc, char* argv[])
{
    return ml::w2v_train(argc, argv);
}

