/***************************************************************************
 * 
 * Copyright (c) 2018aibot.me, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file w2v_util.h
 * @author wangyan(darrenwang00@163.com)
 * @date 2018/04/02 11:51:17
 * @version $Revision$ 
 * @brief word2vec common util
 *  
 **/


#ifndef _ML_W2V_W2V_UTIL_H_
#define _ML_W2V_W2V_UTIL_H_

#include "w2v_define.h"


namespace ml
{

void read_word(char* word, FILE* fin);

long long file_size(char* file_name);

inline int vocab_cmp(const void *a, const void *b)
{
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}


class TimeUsing
{
public:
    explicit TimeUsing() { start(); }
    virtual ~TimeUsing() { }

    void start() { gettimeofday(&_start, NULL); }

    long elapsed()
    {
        gettimeofday(&_end, NULL);
        long using_time = (long) (_end.tv_sec - _start.tv_sec) * (long) 1000000 \
                + (long) (_end.tv_usec - _start.tv_usec);
        return using_time;
    }

	double elapsed_ms() {
		return elapsed() / 1000.0;
	}

private:
    TimeUsing(const TimeUsing &);
    TimeUsing & operator=(const TimeUsing &);
    
    struct timeval _start;
    struct timeval _end;
};


};

#endif  // _ML_W2V_W2V_UTIL_H_

/* vim: set ts=4 sw=4 sts=4 tw=100 */

