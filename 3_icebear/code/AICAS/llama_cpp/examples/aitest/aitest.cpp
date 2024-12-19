#include "common.h"

#include "console.h"
#include "llama.h"
#include "ngram-cache.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <unistd.h>
#include <arm_sve.h>

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static gpt_params               * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;

int main(int argc, char ** argv) {
    gpt_params params;
    g_params = &params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    llama_sampling_params & sparams = params.sparams;
    // struct llama_sampling_context * ctx_sampling = llama_sampling_init(sparams);
    //socket
    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = INADDR_ANY;
    serverAddress.sin_port = htons(9527); // 端口号
    // 绑定地址和端口
    bind(serverSocket, (struct sockaddr *) &serverAddress, sizeof(serverAddress));
    // 监听连接
    listen(serverSocket, 5);
    struct sockaddr_in clientAddress;
    socklen_t clientAddressSize = sizeof(clientAddress);
    int client_socket = accept(serverSocket, (struct sockaddr *) &clientAddress, &clientAddressSize);

    // TODO: Dump params ?
    //LOG("Params perplexity: %s\n", LOG_TOSTR(params.perplexity));

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_TEE("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_TEE("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_TEE("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_TEE("%s: build = %d (%s)\n",      __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG_TEE("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    LOG_TEE("%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    LOG("%s: llama backend init\n", __func__);
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model;
    llama_context * ctx;
    g_model = &model;
    g_ctx = &ctx;

    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    // print system information
    {
        LOG_TEE("\n");
        LOG_TEE("%s\n", get_system_info(params).c_str());
    }

    const bool add_bos = llama_should_add_bos_token(model);
    LOG("add_bos: %d\n", add_bos);

    std::vector<llama_token> embd_inp;    

    llama_ngram_cache ngram_cache_context;
    llama_ngram_cache ngram_cache_dynamic;
    llama_ngram_cache ngram_cache_static;

    // max. number of additional tokens to draft if match is found
    const int n_draft = params.n_draft;

    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    while (1) {
        char buffer[4096] = {0};
        ssize_t bytes_read = recv(client_socket, buffer, sizeof(buffer), 0);
        //设置 prompt 返回 token 数
        if (buffer[0] == '0') {
            char *prompt = buffer + 1;
            params.prompt = prompt;
            embd_inp = ::llama_tokenize(ctx, prompt, add_bos, true);

            {
                // Fill up context ngram cache with tokens from user input:
                const int64_t t_start_draft_us = ggml_time_us();
                llama_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, embd_inp, embd_inp.size(), false);

                if (!params.lookup_cache_static.empty()) {
                    try {
                        ngram_cache_static = llama_ngram_cache_load(params.lookup_cache_static);
                    } catch (std::ifstream::failure const &) {
                        fprintf(stderr, "error: failed to open static lookup cache: %s", params.lookup_cache_static.c_str());
                        exit(1);
                    }
                }

                if (!params.lookup_cache_dynamic.empty()) {
                    try {
                        ngram_cache_dynamic = llama_ngram_cache_load(params.lookup_cache_dynamic);
                    } catch (std::ifstream::failure const &) {} // if the file does not exist it will simply be created at the end of the program
                }
            }

            // LOG("prompt: \"%s\"\n", log_tostr(params.prompt));
            LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

            // Should not run without any tokens
            if (embd_inp.empty()) {
                embd_inp.push_back(llama_token_bos(model));
                LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
            }

            // number of tokens to keep when resetting context
            if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct || params.chatml) {
                params.n_keep = (int)embd_inp.size();
            } else {
                params.n_keep += add_bos; // always keep the BOS token
            }

            if (params.verbose_prompt) {
                LOG_TEE("\n");
                LOG_TEE("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
                LOG_TEE("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
                for (int i = 0; i < (int) embd_inp.size(); i++) {
                    LOG_TEE("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
                }

                if (params.n_keep > add_bos) {
                    LOG_TEE("%s: static prompt based on n_keep: '", __func__);
                    for (int i = 0; i < params.n_keep; i++) {
                        LOG_TEE("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
                    }
                    LOG_TEE("'\n");
                }
                LOG_TEE("\n");
                LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
                LOG_TEE("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
                LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
            }
            int tocken_size = embd_inp.size();
            send(client_socket, &tocken_size, 4, 0);
            // llama_model_load_gemm_weight(model);
        }
        // prefill 测试
        if (buffer[0] == '1') {
            int total_test = buffer[1];
            printf("begin prefill tokens %d test time %d\n", (int)embd_inp.size(), total_test);
            int n_input = embd_inp.size();
            for (int test = 0; test < total_test; test++) {
                llama_kv_cache_clear(ctx);
                llama_decode(ctx, llama_batch_get_one( embd_inp.data(), n_input, 0, 0));
            }
            int tocken_size = embd_inp.size();
            send(client_socket, &tocken_size, 4, 0);
            // llama_model_free_gemm_weight(model);
            // llama_model_load_gemv_weight(model);

            llama_decode(ctx, llama_batch_get_one( embd_inp.data(), n_input - 1, 0,           0));
            llama_decode(ctx, llama_batch_get_one(&embd_inp.back(),           1, n_input - 1, 0));
        }
        // decode 测试
        if (buffer[0] == '2') {
            int total_test = buffer[1];
            int max_new_token = buffer[2];

            printf("begin decode new tokens %d test time %d\n", max_new_token, total_test);
            for (int i = 0; i < total_test; i++) {
                bool input_echo           = true;
                bool display              = true;

                int n_past             = 0;
                int n_remain           = params.n_predict;
                int n_consumed         = 0;

                std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
                std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
                std::ostringstream output_ss;     g_output_ss     = &output_ss;

                // the first thing we will do is to output the prompt, so set color accordingly
                console::set_display(console::prompt);
                display = params.display_prompt;
                params.n_batch = 1;
                std::vector<llama_token> embd(embd_inp.begin(), embd_inp.end());
                int n_predict = 0;

                int n_drafted = 0;
                int n_accept = 0;
                n_predict = 0;
                n_past = embd_inp.size();

                bool has_eos = false;
                std::vector<llama_token> draft;

                llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, 1);
                
                while (n_predict <= max_new_token) {

                    int i_dft = 0;
                    while (true) {
                        // sample from the target model
                        llama_token id = llama_sampling_sample(ctx_sampling, ctx, NULL, i_dft);

                        llama_sampling_accept(ctx_sampling, ctx, id, true);

                        // const std::string token_str = llama_token_to_piece(ctx, id);

                        // if (!params.use_color) {
                        //     printf("%s", token_str.c_str());
                        // }

                        if (id == llama_token_eos(model)) {
                            has_eos = true;
                        }

                        ++n_predict;

                        // check if the target token matches the draft
                        if (i_dft < (int) draft.size() && id == draft[i_dft]) {
                            // LOG("\nthe sampled target token matches the %dth drafted token (%d, '%s') - accepted\n", i_dft, id, token_str.c_str());
                            ++n_accept;
                            ++n_past;
                            ++i_dft;
                            embd.push_back(id);

                            // if (params.use_color) {
                            //     // color accepted draft token
                            //     printf("\033[34m%s\033[0m", token_str.c_str());
                            //     fflush(stdout);
                            // }
                            continue;
                        }

                        // LOG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", id, token_str.c_str());

                        draft.clear();
                        draft.push_back(id);
                        embd.push_back(id);
                        break;
                    }

                    // end of text token
                    if (!embd.empty() && embd.back() == llama_token_eos(model)) {
                        break;
                    }

                    // KV cache management
                    // clean the cache of draft tokens that weren't accepted
                    llama_kv_cache_seq_rm(ctx, 0, n_past, -1);

                    llama_batch_clear(batch_tgt);
                    llama_batch_add(batch_tgt, draft[0], n_past, { 0 }, true);

                    // generate n_pred tokens through prompt lookup
                    auto prompt_lookup = [&]() -> void {
                        const int inp_size = embd.size(); 
                        for (int ngram_size = 2 ; ngram_size > LLAMA_NGRAM_MIN; --ngram_size){
                            const llama_token * ngram = &embd[inp_size - ngram_size];

                            for (int i = 0; i <= (int) inp_size - (ngram_size * 2); ++i) {
                                bool match = true;
                                for (int j = 0; j < ngram_size; ++j) {
                                    if (embd[i + j] != ngram[j]) {
                                        match = false;
                                        break;
                                    }
                                }

                                if (match) {
                                    const int startIdx = i + ngram_size;
                                    const int endIdx = startIdx + n_draft;
                                    if (endIdx < inp_size) {
                                        for (int j = startIdx; j < endIdx; ++j) {
                                            LOG(" - draft candidate %d: %d\n", j, embd[j]);
                                            draft.push_back(embd[j]);
                                            llama_batch_add(batch_tgt, embd[j], n_past + (j - startIdx) + 1, { 0 }, true);
                                            ++n_drafted;
                                        }
                                        return;
                                    }
                                }
                            }
                        }
                        return;
                    };

                    prompt_lookup();

                    llama_decode(ctx, batch_tgt);
                    ++n_past;

                    draft.erase(draft.begin());
                }

                // display text
                if (input_echo && display) {
                    for (auto id : embd) {
                        const std::string token_str = llama_token_to_piece(ctx, id);
                        printf("%s", token_str.c_str());

                        if (embd.size() > 1) {
                            input_tokens.push_back(id);
                        } else {
                            output_tokens.push_back(id);
                            output_ss << token_str;
                        }
                    }
                    fflush(stdout);
                }

                LOG_TEE("\n\n");

                LOG_TEE("\n");
                LOG_TEE("n_draft   = %d\n", n_draft);
                LOG_TEE("n_predict = %d\n", n_predict);
                LOG_TEE("n_drafted = %d\n", n_drafted);

                LOG_TEE("n_accept  = %d\n", n_accept);
                LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

                LOG_TEE("\ntarget:\n");

                while (false && (n_predict <= max_new_token)) {
                    // predict
                    if (!embd.empty()) {
                        // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
                        // --prompt or --file which uses the same value.
                        // evaluate tokens in batches
                        // embd is typically prepared beforehand to fit within a batch, but not always
                        for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                            int n_eval = (int) embd.size() - i;
                            if (n_eval > params.n_batch) {
                                n_eval = params.n_batch;
                            }

                            LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

                            if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                                LOG_TEE("%s : failed to eval\n", __func__);
                                return 1;
                            }

                            n_past += n_eval;

                            LOG("n_past = %d\n", n_past);
                            // Display total tokens alongside total time
                            if (params.n_print > 0 && n_past % params.n_print == 0) {
                                LOG_TEE("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                            }
                        }
                    }

                    embd.clear();

                    if ((int) embd_inp.size() <= n_consumed) {
                        // const llama_token id = llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);
                        const llama_token id = llama_sampling_sample(ctx_sampling, ctx, NULL);

                        llama_sampling_accept(ctx_sampling, ctx, id, true);

                        LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, ctx_sampling->prev).c_str());

                        embd.push_back(id);

                        // echo this to console
                        input_echo = true;

                        // decrement remaining sampling budget
                        --n_remain;

                        LOG("n_remain: %d\n", n_remain);
                    } else {
                        // some user input remains from prompt or interaction, forward it to processing
                        LOG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
                        while ((int) embd_inp.size() > n_consumed) {
                            embd.push_back(embd_inp[n_consumed]);

                            // push the prompt in the sampling context in order to apply repetition penalties later
                            // for the prompt, we don't apply grammar rules
                            llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);

                            ++n_consumed;
                            if ((int) embd.size() >= params.n_batch) {
                                break;
                            }
                        }
                    }

                    ++n_predict;

                    // display text
                    if (input_echo && display) {
                        for (auto id : embd) {
                            const std::string token_str = llama_token_to_piece(ctx, id);
                            printf("%s", token_str.c_str());

                            if (embd.size() > 1) {
                                input_tokens.push_back(id);
                            } else {
                                output_tokens.push_back(id);
                                output_ss << token_str;
                            }
                        }
                        fflush(stdout);
                    }

                    // end of text token
                    if (!embd.empty() && embd.back() == llama_token_eos(model)) {
                        break;
                    }
                }
                
                
                LOG_TEE("\n[end of text]\n");
                llama_kv_cache_clear(ctx);
            }
            send(client_socket, "ok", 2, 0);
            break;
        }
    }

    llama_sampling_free(ctx_sampling);
    // llama_print_timings(ctx);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
