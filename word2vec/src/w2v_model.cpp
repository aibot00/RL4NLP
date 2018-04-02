/***************************************************************************
 * 
 * Copyright (c) 2018aibot.me, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file w2v_model.cpp
 * @author wangyan(darrenwang00@163.com)
 * @date 2018/04/02 11:51:17
 * @version $Revision$ 
 * @brief word2vec model
 *  
 **/
#include "w2v_model.h"

namespace ml
{


//trian model thread
void* train_thread(void* id)
{
    W2VModel* model = W2VModel::get_instance();
    if (model != NULL)
    {
        model->run_thread(id);
    } else {
        std::cerr << "get w2v model error" << std::endl;
    }
    pthread_exit(NULL);
}


//////////////////// W2VModel ////////////////////
W2VModel* W2VModel::_instance = NULL;

W2VModel* W2VModel::get_instance()
{
    if (_instance == NULL)
    {
        _instance = new W2VModel();
        if (_instance == NULL)
        {
            std::cerr << "new W2VModel error!" << std::endl;
            return NULL;
        }
    }
    return _instance;
}


W2VModel::W2VModel()
{
    is_init = false;
}

W2VModel::~W2VModel()
{
    this->destroy();
}

int W2VModel::parser(int argc, char* argv[])
{
    if (argc < 2 || argv == NULL) {
        return RET_FAIL;
    }
    int ret_code = opt.parser_cmd(argc, argv);
    if (ret_code == RET_SUCC) {
        is_init = true;
    }
    return ret_code;
}


int W2VModel::train()
{
    if (!is_init)
    {
        std::cerr << "ERROR: w2v model not init, can not run train" << std::endl;
        return RET_FAIL;
    }

    int i = 0;
    pthread_t *pt = (pthread_t *)malloc(opt.num_threads * sizeof(pthread_t));
    if (pt == NULL) {
        std::cerr << "ERROR: w2v malloc phtread pool error!" << std::endl;
        return RET_FAIL;
    }
    printf("Starting training using file %s\n", opt.train_file);
    
    //init buf
    if (this->init() != RET_SUCC) {
        std::cerr << "ERROR: w2v malloc space error!" << std::endl;
        return RET_FAIL;
    }

    //load vocab
    opt.starting_alpha = opt.alpha; 
    if (opt.read_vocab_file[0] != 0) {
        vocab_dic.load_vocab(opt.read_vocab_file);
    } 
    else {
        vocab_dic.learn_vocab_from_train(opt.train_file);
    }

    if (opt.save_vocab_file[0] != 0) {
        vocab_dic.save_vocab(opt.save_vocab_file);
    }
    if (opt.output_file[0] == 0) {
        std::cerr << "ERROR: w2v the output model name == NULL" << std::endl;
        return -1;
    }

    //init net
    init_net();
    if (opt.negative > 0) {
        this->init_unigram_table();
    }

    //multi run
    start = clock();
    for (i = 0; i < opt.num_threads; ++i) {
        pthread_create(&pt[i], NULL, train_thread, (void *)i);
    }
    for (i = 0; i < opt.num_threads; ++i) {
        pthread_join(pt[i], NULL);
    }

    //save
    this->save();

    //clean
    free(pt);
    pt = NULL;
    return RET_SUCC;
}


int W2VModel::init_unigram_table()
{
    int a, i;
    double train_words_pow = 0;
    double d1, power = 0.75;
    
    table = (int *)malloc(table_size * sizeof(int));
    
    for (a = 0; a < vocab_dic.vocab_size; a++)
    {
        train_words_pow += pow(vocab_dic.vocab[a].cn, power);
    } 

    i = 0;
    d1 = pow(vocab_dic.vocab[i].cn, power) / train_words_pow;

    for (a = 0; a < table_size; a++)
    {
        table[a] = i;
        if (a / (double)table_size > d1) {
            i++;
            d1 += pow(vocab_dic.vocab[i].cn, power) / train_words_pow;
        }
        if (i >= vocab_dic.vocab_size)
        {
            i = vocab_dic.vocab_size - 1;
        }
    }    
    return RET_SUCC;
}


//trian model thread
int W2VModel::run_thread(void* id)
{
    long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    long long local_iter = opt.iter;
    unsigned long long next_random = (long long)id;
    real f, g;
    clock_t now;
    long long layer1_size = opt.layer1_size;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(opt.train_file, "rb");
    if (fi == NULL) {
        std::cerr << "open train file error!" << std::endl;
        return RET_FAIL;
    }
    
    long thread_offset = opt.file_size / (long long)opt.num_threads * (long long)id;

    fseek(fi, thread_offset, SEEK_SET);

    while (1)
    {
        //print pro and update alpha
        if (word_count - last_word_count > 10000)
        {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((opt.debug_mode > 1))
            {
                now = clock();
                printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, opt.alpha,
                    word_count_actual / (real)(opt.iter * opt.train_words + 1) * 100,
                    word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }

            opt.alpha = opt.starting_alpha * \
                    (1 - word_count_actual / (real)(opt.iter * opt.train_words + 1));
            if (opt.alpha < opt.starting_alpha * 0.0001) {    
                opt.alpha = opt.starting_alpha * 0.0001;
            }
        }
        
        //read a sentence and random cutfoff frequent words
        if (sentence_length == 0)
        {
            while (1) 
            {
                word = read_word_index(fi);

                if (feof(fi)) break;
                if (word == -1) continue;
                word_count++;
                
                if (word == 0) break;
                
                // The subsampling randomly discards frequent words while keeping the ranking same
                if (opt.sample > 0)
                {
                    real ran = (sqrt(vocab_dic.vocab[word].cn / (opt.sample * opt.train_words)) + 1) * (opt.sample * opt.train_words) / vocab_dic.vocab[word].cn;
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                }
                sen[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }

        //check thread end and update iter num
        if (feof(fi) || (word_count > opt.train_words / opt.num_threads))
        {
            word_count_actual += word_count - last_word_count;
            local_iter--;
            if (local_iter == 0) break;
            word_count = 0;
            last_word_count = 0;
            sentence_length = 0;
            fseek(fi, thread_offset, SEEK_SET);
            continue;
        }

        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < layer1_size; c++) {
            neu1[c] = 0;
            neu1e[c] = 0;
        }
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % opt.window;

        if (opt.cbow) 
        {  //train the cbow architecture
            // in -> hidden
            cw = 0;
            for (a = b; a < opt.window * 2 + 1 - b; a++) if (a != opt.window)
            {
                c = sentence_position - opt.window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                
                last_word = sen[c];
                if (last_word == -1) continue;
                for (c = 0; c < layer1_size; c++) {
                    neu1[c] += syn0[c + last_word * layer1_size];
                }                
                cw++;
            }

            if (cw)
            {
                for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
                if (opt.hs)
                {
                    for (d = 0; d < vocab_dic.vocab[word].codelen; d++)
                    {
                        f = 0;
                        l2 = vocab_dic.vocab[word].point[d] * layer1_size;
                        // Propagate hidden -> output
                        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
                        if (f <= -MAX_EXP) continue;
                        else if (f >= MAX_EXP) continue;
                        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        // 'g' is the gradient multiplied by the learning rate
                        g = (1 - vocab_dic.vocab[word].code[d] - f) * opt.alpha;
                        // Propagate errors output -> hidden
                        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                        // Learn weights hidden -> output
                        for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
                    }
                }

                // NEGATIVE SAMPLING
                if (opt.negative > 0)
                {
                    for (d = 0; d < opt.negative + 1; d++)
                    {
                        if (d == 0) {
                            target = word;
                            label = 1;
                        } else {
                            next_random = next_random * (unsigned long long)25214903917 + 11;
                            target = table[(next_random >> 16) % table_size];
                            if (target == 0) target = next_random % (vocab_dic.vocab_size - 1) + 1;
                            if (target == word) continue;
                            label = 0;
                        }

                        l2 = target * layer1_size;
                        f = 0;
                        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
                        if (f > MAX_EXP) g = (label - 1) * opt.alpha;
                        else if (f < -MAX_EXP) g = (label - 0) * opt.alpha;
                        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * opt.alpha;
                        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
                    }
                }

                // hidden -> in
                for (a = b; a < opt.window * 2 + 1 - b; a++) if (a != opt.window)
                {
                    c = sentence_position - opt.window + a;
                    if (c < 0) continue;
                    if (c >= sentence_length) continue;
                    last_word = sen[c];
                    if (last_word == -1) continue;
                    for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
                }
            }
        }
        else {  //train skip-gram
            for (a = b; a < opt.window * 2 + 1 - b; a++) if (a != opt.window)
            {
                c = sentence_position - opt.window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                last_word = sen[c];
                if (last_word == -1) continue;
                l1 = last_word * layer1_size;
                for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

                // HIERARCHICAL SOFTMAX
                if (opt.hs) 
                for (d = 0; d < vocab_dic.vocab[word].codelen; d++) {
                    f = 0;
                    l2 = vocab_dic.vocab[word].point[d] * layer1_size;
                    // Propagate hidden -> output
                    for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
                    if (f <= -MAX_EXP) continue;
                    else if (f >= MAX_EXP) continue;
                    else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                    // 'g' is the gradient multiplied by the learning rate
                    g = (1 - vocab_dic.vocab[word].code[d] - f) * opt.alpha;
                    // Propagate errors output -> hidden
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                    // Learn weights hidden -> output
                    for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
                }

                // NEGATIVE SAMPLING
                if (opt.negative > 0) 
                for (d = 0; d < opt.negative + 1; d++)
                {
                    if (d == 0) {
                        target = word;
                        label = 1;
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        target = table[(next_random >> 16) % table_size];
                        if (target == 0) target = next_random % (vocab_dic.vocab_size - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }

                    l2 = target * layer1_size;
                    f = 0;
                    for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
                    if (f > MAX_EXP) g = (label - 1) * opt.alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * opt.alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * opt.alpha;
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                    for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
                } // end of for d

                // Learn weights input -> hidden
                for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
            }
        }

        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    
    return 0;
}



int W2VModel::save()
{
    long a, b, c, d;
    long layer1_size = opt.layer1_size;
    long vocab_size = vocab_dic.vocab_size;
    FILE* fo = fopen(opt.output_file, "wb");

    if (opt.classes == 0) 
    {
        // Save the word vectors
        fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
        
        for (a = 0; a < vocab_size; a++)
        {
            fprintf(fo, "%s ", vocab_dic.vocab[a].word);
            if (opt.binary) {
                 for (b = 0; b < layer1_size; b++) {
                     fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
                 }
            }
            else {
                for (b = 0; b < layer1_size; b++) {
                     fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
                }
            }
            fprintf(fo, "\n");
        }
    } 
    
    else {
        // Run K-means on the word vectors
        int clcn = opt.classes, iter = 10, closeid;
        int *centcn = (int *)malloc(opt.classes * sizeof(int));
        int *cl = (int *)calloc(vocab_size, sizeof(int));
        real closev, x;
        real *cent = (real *)calloc(opt.classes * layer1_size, sizeof(real));
        for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;

        for (a = 0; a < iter; a++)
        {
            for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
            for (b = 0; b < clcn; b++) centcn[b] = 1;
            for (c = 0; c < vocab_size; c++) 
            {
                for (d = 0; d < layer1_size; d++) {
                    cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
                }            
                centcn[cl[c]]++;
            }

            for (b = 0; b < clcn; b++)
            {
                closev = 0;
                for (c = 0; c < layer1_size; c++) {
                    cent[layer1_size * b + c] /= centcn[b];
                    closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
                }
                closev = sqrt(closev);
                for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
            }

            for (c = 0; c < vocab_size; c++)
            {
                closev = -10;
                closeid = 0;
                for (d = 0; d < clcn; d++)
                {
                    x = 0;
                    for (b = 0; b < layer1_size; b++) {
                        x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
                    }
                    if (x > closev)
                    {
                        closev = x;
                        closeid = d;
                    }
                }
                cl[c] = closeid;
            }
        }

        // Save the K-means classes
        for (a = 0; a < vocab_size; a++) {
            fprintf(fo, "%s %d\n", vocab_dic.vocab[a].word, cl[a]);
        }

        //clean
        free(centcn);
        free(cent);
        free(cl);
    }

    fclose(fo);

    return RET_SUCC;
}



int W2VModel::destroy()
{
    if (syn0 != NULL) {
        free(syn0);
        syn0 = NULL;
    }
    if (syn1 != NULL) {
        free(syn1);
        syn1 = NULL;
    }
    if (syn1neg != NULL) {
        free(syn1neg);
        syn1neg = NULL;
    }
    if (expTable != NULL) {
        free(expTable);
        expTable = NULL;
    }

    return RET_SUCC;
}


int W2VModel::init()
{
    int i = 0;

    vocab_dic.init(&opt);

    //init exptable
    //sigmoid 函数查表： 1~1000 对应到 -6 ~ 6
    //        对应关系： i -> i* 6*2 /1000 - 6
    // 1 / (1+e^-x) = e^x /(e^x + 1)
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++)
    {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }

    return RET_SUCC;
}


int W2VModel::init_net()
{
    long long a, b;
    unsigned long long next_random = 1;

    long long layer1_size = opt.layer1_size;
    long long vocab_size = vocab_dic.vocab_size; 

    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}

    if (opt.hs) {
        a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
        if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
        syn1[a * layer1_size + b] = 0;
    }

    if (opt.negative > 0)
    {
        a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
        if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] = 0;
    }

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
    {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }

    return create_huffman_tree();
}



// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
int W2VModel::create_huffman_tree()
{
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
    long long vocab_size = vocab_dic.vocab_size; 

    long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

    for (a = 0; a < vocab_size; a++) count[a] = vocab_dic.vocab[a].cn;
    for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;

    pos1 = vocab_size - 1;
    pos2 = vocab_size;
    // Following algorithm constructs the Huffman tree by adding one node at a time
    for (a = 0; a < vocab_size - 1; a++)
    {
        // First, find two smallest nodes 'min1, min2'
        if (pos1 >= 0)
        {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                pos1--;
            } else {
                min1i = pos2;
                pos2++;
            }
        } else {
            min1i = pos2;
            pos2++;
        }

        if (pos1 >= 0)
        {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                pos1--;
            } else {
                min2i = pos2;
                pos2++;
            }
        } else {
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + a;
        parent_node[min2i] = vocab_size + a;
        binary[min2i] = 1;
    }

    // Now assign binary code to each vocabulary word
    for (a = 0; a < vocab_size; a++)
    {
        b = a;
        i = 0;
        while (1) {
            code[i] = binary[b];
            point[i] = b;
            i++;
            b = parent_node[b];
            if (b == vocab_size * 2 - 2) break;
        }

        vocab_dic.vocab[a].codelen = i;
        vocab_dic.vocab[a].point[0] = vocab_size - 2;
        for (b = 0; b < i; b++) {
            vocab_dic.vocab[a].code[i - b - 1] = code[b];
            vocab_dic.vocab[a].point[i - b] = point[b] - vocab_size;
        }
    }

    free(count);
    free(binary);
    free(parent_node);
    return RET_SUCC;
}


};

/* vim: set ts=4 sw=4 sts=4 tw=100 */

