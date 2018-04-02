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


#include "w2v_model.h"


///// mian /////
int main(int argc, char* argv[])
{
    ml::W2VModel* m = ml::W2VModel::get_instance();
    m->parser(argc, argv);
    m->train();
    
    delete m;
    m = NULL;

    std::cout << "done" << std::endl;
    return 0;
}


