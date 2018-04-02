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
#include "w2v_util.h"


namespace ml
{


//read one word from file
//space + tab + EOL to be word boundaries
void read_word(char* word, FILE* fin)
{
    int i = 0, ch;
    while (!feof(fin))
    {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
        {
            if (i > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            } else {
                continue;
            }
        }
        word[i] = ch;
        i++;
        if (i >= MAX_STRING - 1) i--;   // Truncate too long words
    }
    word[i] = 0;
}



long long file_size(char* file_name)
{
    long long size = 0;
    FILE* fin = fopen(file_name, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        return size;
    }

    fseek(fin, 0, SEEK_END);
    size = ftell(fin);
    fclose(fin);
    return size;
}


}; // end of namespace ml

/* vim: set ts=4 sw=4 sts=4 tw=100 */

