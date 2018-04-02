/***************************************************************************
 * 
 * Copyright (c) 2018aibot.me, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file w2v_option.cpp
 * @author wangyan(darrenwang00@163.com)
 * @date 2018/04/02 11:51:16
 * @version $Revision$ 
 * @brief option
 *  
 **/
#include "w2v_option.h"


namespace ml
{


W2VOption::W2VOption()
{
    this->reset();
}

W2VOption::~W2VOption()
{
  this->reset();
}

int W2VOption::reset()
{
    //file
    train_file[0] = 0;
    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;

    //param
    hs = 0;
    cbow = 1;
    window = 5;
    iter = 5;
    negative = 5;
    layer1_size = 100;
    alpha = 0.025;
    sample = 1e-3;
    starting_alpha = 0.0;
    binary = 0;
    debug_mode = 2;
    min_count = 5;
    num_threads = 12;
    min_reduce = 1;
    classes = 0;

    //mid data
    train_words = 0;
    file_size = 0;
    return RET_SUCC;
}

int W2VOption::help_info()
{
    std::ostringstream os;
    os << "WORD VECTOR estimation toolkit v 0.1c\n\n";
    os << "Options:\n";
    os << "Parameters for training:\n";
    os << "\t-train <file>\n";
    os << "\t\tUse text data from <file> to train the model\n";
    os << "\t-output <file>\n";
    os << "\t\tUse <file> to save the resulting word vectors / word clusters\n";
    os << "\t-size <int>\n";
    os << "\t\tSet size of word vectors; default is 100\n";
    os << "\t-window <int>\n";
    os << "\t\tSet max skip length between words; default is 5\n";
    os << "\t-sample <float>\n";
    os << "\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n";
    os << "\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n";
    os << "\t-hs <int>\n";
    os << "\t\tUse Hierarchical Softmax; default is 0 (not used)\n";
    os << "\t-negative <int>\n";
    os << "\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n";
    os << "\t-threads <int>\n";
    os << "\t\tUse <int> threads (default 12)\n";
    os << "\t-iter <int>\n";
    os << "\t\tRun more training iterations (default 5)\n";
    os << "\t-min-count <int>\n";
    os << "\t\tThis will discard words that appear less than <int> times; default is 5\n";
    os << "\t-alpha <float>\n";
    os << "\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n";
    os << "\t-classes <int>\n";
    os << "\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n";
    os << "\t-debug <int>\n";
    os << "\t\tSet the debug mode (default = 2 = more info during training)\n";
    os << "\t-binary <int>\n";
    os << "\t\tSave the resulting vectors in binary moded; default is 0 (off)\n";
    os << "\t-save-vocab <file>\n";
    os << "\t\tThe vocabulary will be saved to <file>\n";
    os << "\t-read-vocab <file>\n";
    os << "\t\tThe vocabulary will be read from <file>, not constructed from the training data\n";
    os << "\t-cbow <int>\n";
    os << "\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n";
    os << "\nExamples:\n";
    os << "./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n";
    std::cerr << os.str() << std::endl;
    return RET_SUCC;
}


int W2VOption::parser_cmd(int argc, char **argv)
{
    int i = 0;
    if (argc == 1) {
        help_info();
        return -1;
    }

    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;

    if ((i = parse_one((char *)"-size", argc, argv)) > 0) {
        layer1_size = atoi(argv[i + 1]);
    }
    if ((i = parse_one((char *)"-train", argc, argv)) > 0) {
        strcpy(train_file, argv[i + 1]);
    }
    if ((i = parse_one((char *)"-save-vocab", argc, argv)) > 0) {
        strcpy(save_vocab_file, argv[i + 1]);
    }
    if ((i = parse_one((char *)"-read-vocab", argc, argv)) > 0) {
        strcpy(read_vocab_file, argv[i + 1]);
    }
    if ((i = parse_one((char *)"-debug", argc, argv)) > 0) {
        debug_mode = atoi(argv[i + 1]);
    }

    if ((i = parse_one((char *)"-binary", argc, argv)) > 0) {
        binary = atoi(argv[i + 1]);
    }
    if ((i = parse_one((char *)"-cbow", argc, argv)) > 0) {
        cbow = atoi(argv[i + 1]);
    } 
    if (cbow) alpha = 0.05;

    if ((i = parse_one((char *)"-alpha", argc, argv)) > 0) {
        alpha = atof(argv[i + 1]);
    }

    if ((i = parse_one((char *)"-output", argc, argv)) > 0) {
        strcpy(output_file, argv[i + 1]);
    }

    if ((i = parse_one((char *)"-window", argc, argv)) > 0) {
        window = atoi(argv[i + 1]);
    }
    if ((i = parse_one((char *)"-sample", argc, argv)) > 0) {
        sample = atof(argv[i + 1]);
    }

    if ((i = parse_one((char *)"-hs", argc, argv)) > 0) {
        hs = atoi(argv[i + 1]);
    }

    if ((i = parse_one((char *)"-negative", argc, argv)) > 0) {
        negative = atoi(argv[i + 1]);
    }

    if ((i = parse_one((char *)"-threads", argc, argv)) > 0) {
        num_threads = atoi(argv[i + 1]);
    }

    if ((i = parse_one((char *)"-iter", argc, argv)) > 0) {
        iter = atoi(argv[i + 1]);
    }
    if ((i = parse_one((char *)"-min-count", argc, argv)) > 0) {
        min_count = atoi(argv[i + 1]);
    }
    if ((i = parse_one((char *)"-classes", argc, argv)) > 0) {
        classes = atoi(argv[i + 1]);
    }
    
    std::cout << this->debug() << std::endl;
    return RET_SUCC;
}


int W2VOption::parse_one(char *str, int argc, char **argv)
{
    if (str == NULL || argv == NULL) {
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(str, argv[i]))
        {
            if (i == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}


std::string W2VOption::debug()
{
    std::ostringstream os;
    os << "train_file= " << train_file << std::endl;
    os << "output_file= " << output_file << std::endl;
    return os.str();
}


}; // end of namespace ml

/* vim: set ts=4 sw=4 sts=4 tw=100 */
