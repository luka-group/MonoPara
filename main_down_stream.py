# Created by Qin Liu on 2023/12/30
# Mostly copied from xunannancy
import argparse
import numpy as np
import os
import gc
from datasets import load_dataset
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WORLD_SIZE"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import json
# from repositories.SimCTG.document_generation.inference_dataclass import Data
from tqdm import tqdm
from repositories.SimCTG.simctg.evaluation import measure_repetition_and_diversity
import torch
# from klgreedy import KLGreedy
from repositories.SimCTG.document_generation.simctg import SimCTG
from repositories.SimCTG.document_generation.utlis import enlarge_past_key_values
import torch.nn as nn
import torch.nn.functional as F
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, LlamaForCausalLM
from evaluate import load
import math
import rouge
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import BERTScorer
from mannual_insts import LABEL_SET, mannual_instructions
from OOD_insts import OOD_clean_instructions
from chatgpt_instructs import chatgpt_insts
from vague_instructs import long_insts
from search_based_long_para import search_based_long
from peft import PeftModel
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

# from poc_coherence import dataset_dict


def read_dataset(dataset_path):
    # Extracting the relevant parts from the provided data format
    prompts = list()
    # ToDo: read the dataset
    data = None
    # Splitting the data into question and answer parts
    parts = data.split("\n")
    # Extracting only the question part
    question = parts[0].split("Q: ")[1]
    promts.append(question)

    return prompts


def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam//beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))    # [B, K, num_head, seq_len, esz]
            item = item[range(bsz), selected_idx, :, :, :]   # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values

def instance_past_key_values(past_key_values):
    new_key_values = []
    for layer_idx, layer in enumerate(past_key_values):
        for idx, item in enumerate(layer):
            bsz, num_head, seq_len, esx = item.size()
            if idx == 0:
                items = [[item[i].unsqueeze(dim=0)] for i in range(bsz)]
            else:
                for i in range(bsz):
                    items[i].append(item[i].unsqueeze(dim=0))
        if layer_idx == 0:
            new_key_values = [[bsz] for bsz in items]
        else:
            for bsz in range(len(items)):
                new_key_values[bsz].append(items[bsz])

    return new_key_values

def measure_ppl(sent, model_name, perplexity):
    with torch.no_grad():
        ppl = perplexity.compute(predictions=[sent], model_id=model_name, device='cuda')['perplexities']
    return torch.tensor(ppl)

def KLGreedyDecodingOneStep(model, ids, gen_ids, beam_width, prefix_seqlen, past_key_values, last_logits_logsoftmax,
                            option, tokenizer, alpha):
    device = model.device
    output = model(
        input_ids=ids,
        past_key_values=past_key_values,
        use_cache=True,
        attention_mask=torch.ones([1, 1+past_key_values[0][0].shape[-2]], dtype=torch.bool, device=device) if past_key_values is not None else None
    )
    past_key_values = output.past_key_values
    past_key_values_per_instance = instance_past_key_values(past_key_values)
    bsz = ids.shape[0]
    assert bsz == 1
    logit_for_next_step = output.logits[:, -1, :]    # [B, V]

    if last_logits_logsoftmax is None:
        next_id = torch.argmax(logit_for_next_step, dim=-1).unsqueeze(-1)
        last_logits_logsoftmax = F.log_softmax(output.logits, dim=-1)
        if 'opt' in model.name_or_path:
            last_logits_logsoftmax = last_logits_logsoftmax[:, 1:, :]  # ignore the </s> to avoid prefix-kl repeting </s>
        prefix_seqlen = last_logits_logsoftmax.shape[1]
        gen_ids = next_id.squeeze(-1)
    else:
        logits_logsoftmax = F.log_softmax(logit_for_next_step, dim=-1)  # [B, V]
        _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)  # [B, K]
        # kl_loss = nn.KLDivLoss(reduction='none', log_target=True)
        next_id_list = list()
        updated_logits_logsoftmax = torch.cat([last_logits_logsoftmax, logits_logsoftmax.unsqueeze(dim=1)], dim=1)  # [B, L+1, V]
        used_last_logits_logsoftmax = last_logits_logsoftmax

        past_seqlen = used_last_logits_logsoftmax.shape[1]
        for i in range(bsz):
            # kl_list = kl_loss(logits_logsoftmax[i].repeat(past_seqlen, 1), used_last_logits_logsoftmax[i]).sum(dim=(-1,))
            next_id = torch.argmax(logit_for_next_step[i])

            # random from top k
            if 'include' in option: #option.startswith('include'):
                candidates = top_k_ids[i]
            elif 'exclude' in option: #.startswith('exclude'):
                # min_idx = torch.argmin(kl_list)
                # min_token = torch.argmax(last_logits_logsoftmax[i][min_idx])
                # candidates = top_k_ids[i][top_k_ids[i] != min_token]
                candidates = top_k_ids[i][top_k_ids[i] != next_id]
            # enlarged_past_key_values = enlarge_past_key_values(past_key_values_per_instance[i], len(candidates))
            # output = model(
            #     input_ids=candidates.view(-1, 1),
            #     attention_mask=torch.ones([beam_width, 1+past_key_values[0][0].shape[-2]], dtype=torch.bool, device=device), #torch.ones_like(candidates.view(-1, 1)),
            #     past_key_values=enlarged_past_key_values,
            #     use_cache=True,
            # )
            # logp = F.log_softmax(output.logits, dim=-1) # (batch_size, sequence_length, config.vocab_size)
            ppl_list = list()

            if "discrete" in option:
                with torch.no_grad():
                    for j in range(len(candidates)):
                        tmp_gen_ids = torch.cat([gen_ids, candidates[j].unsqueeze(-1)], dim=0).unsqueeze(0)
                        tmp_gen_ids.to('cuda')
                        output = model(input_ids=tmp_gen_ids, labels=tmp_gen_ids)
                        ppl_list.append(torch.exp(output.loss))
                    ppl_list = torch.stack(ppl_list) # [#candidates]
                if 'exclude_min' in option:
                    next_id_new = candidates[torch.argmin(ppl_list)]
                else:
                    next_id_new = candidates[torch.multinomial(input=torch.softmax(-ppl_list, dim=0), num_samples=1)[0]]

            else:
                with torch.no_grad():
                    target_output = model(input_ids=gen_ids.unsqueeze(0), use_cache=True)
                    logit_for_next_step_target = target_output.logits[:, -1, :]
                    ensembled_logits = logit_for_next_step[i] * (1. - alpha) + logit_for_next_step_target[i] * alpha
                    next_id_new = torch.argmax(ensembled_logits)

            next_id_list.append(next_id_new)
            if option.endswith('_switch'):
                old, new = torch.clone(logits_logsoftmax[i][next_id]), torch.clone(logits_logsoftmax[i][next_id_new])
                logits_logsoftmax[i][next_id] = new
                logits_logsoftmax[i][next_id_new] = old

        next_id = torch.stack(next_id_list)
        gen_ids = torch.cat([gen_ids, next_id], dim=0)
        next_id = next_id.unsqueeze(-1)
        last_logits_logsoftmax = torch.cat([last_logits_logsoftmax, logits_logsoftmax.unsqueeze(1)], dim=1)    # [B, S, E]

    return next_id, past_key_values, last_logits_logsoftmax, prefix_seqlen, gen_ids


class KLGreedy(SimCTG):
    def __init__(self, model_name, pad_token_id):
        super().__init__(model_name, pad_token_id)

    def ppl_greedy(self, input_ids, beam_width, decoding_len, option, tokenizer, alpha):
        self.model.eval()

        generated = [item for item in input_ids.tolist()]
        past_key_values = None
        last_logits_logsoftmax = None
        prefix_seqlen = None
        gen_ids = None
        for step in range(decoding_len):
            input_ids, past_key_values, last_logits_logsoftmax, prefix_seqlen, gen_ids = KLGreedyDecodingOneStep(
                self.model,
                input_ids,
                gen_ids,
                beam_width,
                prefix_seqlen,
                past_key_values,
                last_logits_logsoftmax,
                option,
                tokenizer,
                alpha
            )
            tokens = input_ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)
            if generated[0][-1] == tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]:
                break
        return generated[0]

    def eval_ppl(self, input_ids):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids=input_ids, labels=input_ids)
            ppl = torch.exp(output.loss)
        return ppl

def run_PPL_greedy(dataset, model_name, k, option_, device_id, iter, decoding_len=128, alpha=0.5):
    print("alpha: ", alpha)

    device = torch.device(f'cuda:{device_id}')
    option = f'L{decoding_len}_{option_}'

    # doc_save_path = f'./ppl_sort_greedy_gen/{model_name}/{option}/gen_doc.json'
    # os.makedirs(os.path.dirname(doc_save_path), exist_ok=True)

    print('Loading pre-trained model...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.bos_token])[0]
    model = KLGreedy(model_name, pad_token_id)
    model = model.to(device)
    model.eval()
    print('Model loaded')

    print('----------------------------------------------------------------')
    print('Start inference...')
    # test_num = len(data.split_prefix_token_id_list)
    result_list = []

    with torch.no_grad():
        for prompt in tqdm(dataset, total=len(dataset)):
            # print("data:", data)
            if "llama" in model_name:
                data = f"Generate one paraphrase of the following sentence. Just paraphrase, no other words.\n{prompt}"
            elif "alpha" in model_name:
                data = f"GPT4 Correct User: Please provide just one paraphrase for the following sentence: '{prompt}'.<|end_of_turn|>GPT4 Correct Assistant:"
            else:
                data = f"[INST] Generate one paraphrase of the following sentence. Just paraphrase, no other words.\n{prompt} [/INST]"
            input_ids = tokenizer.encode(data, return_tensors='pt').to(device)
            prefix_len = input_ids.shape[-1]
            one_res_ids = model.ppl_greedy(input_ids, k, decoding_len, option, tokenizer, alpha)
            one_res_sent_ids = one_res_ids[prefix_len:-1]
            whole_sent = tokenizer.decode(one_res_ids)
            paraphrase_sent = tokenizer.decode(one_res_sent_ids)

            # print("==== generation result ====")
            # print("whole sentence: ", whole_sent)
            print("***** paraphrase sentence: *****")
            print(paraphrase_sent)

            # calculate PPL
            loss = model.model(torch.tensor(one_res_sent_ids).unsqueeze(0).to("cuda"), labels=torch.tensor(one_res_sent_ids).unsqueeze(0).to("cuda")).loss
            ppl = torch.exp(loss)
            print("PPL of paraphrase: ", ppl.item())

            result_list.append({"ori_prompt": prompt, "paraphrase": paraphrase_sent, "ppl": ppl.item(), "iter": iter})

    # with open("circulation.json", 'a', encoding="utf-8") as file:
    #     json.dump(result_list, file, ensure_ascii=False, indent=4)

    del model
    # torch.cuda.empty_cache()
    gc.collect()
    return result_list, [paraphrase_sent], iter + 1, ppl.item()


def eval_PPL(text, args, model=None, tokenizer=None):
    if "llamakkk" in args.model_name:
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            "tloen/alpaca-lora-7b",
            torch_dtype=torch.float16,
        )
        model.to('cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encodeds = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    model_inputs = encodeds.to(model.device)
    loss = model(**model_inputs, labels=model_inputs['input_ids']).loss
    ppl = torch.exp(loss)
    del model
    # torch.cuda.empty_cache()
    gc.collect()
    return ppl.item()


def eval_downstream(prompt, dataset, last_sent, label_domain, model, tokenizer, args):
    print("prompt: ", prompt)
    acc = 0
    count = 0

    if "llamakkk" in args.model_name:
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            "tloen/alpaca-lora-7b",
            torch_dtype=torch.float16,
        )
        model.to('cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to('cuda')

    with torch.no_grad():
        # for idx in tqdm(range(len(dataset["sentence"]))):
        for idx in tqdm(range(len(dataset[args.content_name]))):
        # for idx in tqdm(range(5)):
            data = dataset[args.content_name][idx]
            if args.dataset_name == "gimmaru/newspop":
                label = dataset['topic'][idx]
            else:
                label = dataset['label'][idx]
            if "llama" in args.model_name:
                input = data + "\n" + prompt + " " + last_sent
            elif "alpha" in args.model_name:
                input = f"GPT4 Correct User: {data}\n{prompt} {last_sent}.<|end_of_turn|>GPT4 Correct Assistant:"
            else:
                # input = "[INST]" + prompt + "\n" + data + " " + last_sent + " [/INST]"
                input = "[INST]" + data + "\n" + prompt + " " + last_sent + " [/INST]"
            encodeds = tokenizer(input, return_tensors="pt", add_special_tokens=False)
            model_inputs = encodeds.to(model.device)
            output = model(**model_inputs, labels=model_inputs['input_ids'])
            logit_for_next_step = output.logits[:, -1, :].to('cpu')
            label_domain = torch.tensor(label_domain)
            pred_logits = logit_for_next_step[:, label_domain]
            if args.dataset_name == "gimmaru/newspop":
                label_id = tokenizer.encode(label, return_tensors="pt")
                if label_domain[torch.argmax(pred_logits)] == label_id[0][1]:
                    acc += 1
                count += 1
            else:
                pred_label = torch.argmax(pred_logits)
                if pred_label == label:
                    acc += 1
                count += 1

    del model
    # torch.cuda.empty_cache()
    gc.collect()

    return acc / count


# def eval_avg_instance_ppl(prompt, dataset, model, tokenizer, args):
#     num = 0
#     sum_ppl = 0.
#     print("evaluate avg instance ppl of prompt: ", prompt)
#     for sample in tqdm(dataset):
#         data = sample['text']
#         # input = data + "\n" + prompt
#         input = prompt + "\n" + data
#         encodeds = tokenizer(input, return_tensors="pt", add_special_tokens=False)
#         model_inputs = encodeds.to(model.device)
#         loss = model(**model_inputs, labels=model_inputs['input_ids']).loss
#         ppl = torch.exp(loss)
#         sum_ppl += ppl.item()
#         num += 1
#     return sum_ppl / num


def eval_avg_instance_ppl(prompt, dataset, model, tokenizer, args):
    if "llamakkk" in args.model_name:
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            "tloen/alpaca-lora-7b",
            torch_dtype=torch.float16,
        )
        model.to('cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to('cuda')
    num = 0
    sum_ppl = 0.
    print("evaluate avg instance ppl of prompt: ", prompt)

    labels = LABEL_SET[args.dataset_name]
    last_sent = "Choices: " + ", ".join(labels) + ". Answer with one word:"

    with torch.no_grad():
        for sample in tqdm(dataset):
            data = sample[args.content_name]
            if "llama" in args.model_name:
                input = data + "\n" + prompt + " " + last_sent
            elif "alpha" in args.model_name:
                input = f"GPT4 Correct User: {data}\n{prompt} {last_sent}.<|end_of_turn|>GPT4 Correct Assistant:"
            else:
                input = "[INST]" + data + "\n" + prompt + " " + last_sent + " [/INST]"
            encodeds = tokenizer(input, return_tensors="pt", add_special_tokens=False)
            model_inputs = encodeds.to(model.device)
            loss = model(**model_inputs, labels=model_inputs['input_ids']).loss
            ppl = torch.exp(loss)
            sum_ppl += ppl.item()
            num += 1

    del model
    # torch.cuda.empty_cache()
    gc.collect()

    return sum_ppl / num


def eval_paraphrase(ori, para):
    # func = nltk.translate.bleu_score.SmoothingFunction()
    # ori_1 = nltk.word_tokenize(ori.strip().lower())
    # para_1 = nltk.word_tokenize(para.strip().lower())
    # bleu = sentence_bleu([ori_1], para_1, weights=(0.5, 0.5), smoothing_function=func.method1)
    # meteor = meteor_score([ori_1], para_1)
    # rougeL = rouge.Rouge(metrics=['rouge-l']).get_scores(ori, para)[0]['rouge-l']['f']
    # rouge1 = rouge.Rouge(metrics=['rouge-1']).get_scores(ori, para)[0]['rouge-1']['f']
    # rouge2 = rouge.Rouge(metrics=['rouge-2']).get_scores(ori, para)[0]['rouge-2']['f']

    bertscore = load("bertscore")
    bert_result = bertscore.compute(predictions=[para], references=[ori], lang="en")

    # meteor_scorer = load('meteor')
    # meteor_1 = meteor_scorer.compute(predictions=[para], references=[ori])

    # from repositories.BARTScore.bart_score import BARTScorer
    # bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
    # bart_result = bart_scorer.score([ori], [ori], batch_size=4)  # generation scores from the first list of texts to the second list of texts.

    return bert_result["f1"][0]


def greedy_gen(ori, args):
    if "llamakkk" in args.model_name:
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            # load_in_8bit=True,
            # torch_dtype=torch.float16,
            # device_map="auto",
        )
        # model = prepare_model_for_int8_training(model)
        # # model = PeftModel.from_pretrained(
        # #     model,
        # #     "tloen/alpaca-lora-7b",
        # #     torch_dtype=torch.float16,
        # # )
        # config = LoraConfig(
        #     r=16,
        #     lora_alpha=16,
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        #     lora_dropout=0.05,
        #     inference_mode=True,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        # )
        # model = get_peft_model(model, config)
        # adapters_weights = torch.load("/nas02/qinliu/PPL_gen/generation_loss_only/adapter_model.bin",
        #                               map_location='cuda:0')
        # set_peft_model_state_dict(model, adapters_weights)
        model.to('cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if "llama" in args.model_name:
        instruction = f"Generate one paraphrase of the following sentence. Just paraphrase, no other words.\n{ori}"
    elif "alpha" in args.model_name:
        instruction = f"GPT4 Correct User: Please provide just one paraphrase for the following sentence: '{ori}'.<|end_of_turn|>GPT4 Correct Assistant:"
    else:
        instruction = f"[INST] Generate one paraphrase of the following sentence. Just paraphrase, no other words.\n{ori} [/INST]"
    input_ids = tokenizer.encode(instruction, return_tensors="pt").to('cuda')
    num_tokens = len(input_ids[0])
    mask = torch.where(input_ids == model.config.eos_token_id, 0, 1).to('cuda')
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=mask,
        do_sample=False,  # greedy search
        num_beams=1,  # greedy search
        pad_token_id=model.config.eos_token_id,
        max_new_tokens=30,
        return_dict_in_generate=True)
    output_ids = outputs.sequences.to('cpu')
    output_prompt = tokenizer.decode(output_ids[0, num_tokens:], skip_special_tokens=True)

    encodeds = tokenizer(output_prompt, return_tensors="pt", add_special_tokens=False)
    model_inputs = encodeds.to('cuda')
    loss = model(**model_inputs, labels=model_inputs['input_ids']).loss
    ppl = torch.exp(loss)

    del model
    # torch.cuda.empty_cache()
    gc.collect()

    return output_prompt, ppl.item()


if __name__ == '__main__':
    # python KLGreedySort_gridsearch.py -dataset wikitext103 -m gpt2 -k 5 -alpha 1.6 -id -1 -w -1
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-data_path', type=str, default='./data/inputs.jsonl')
    parser.add_argument('--model_name', '-m', type=str, help='model_name', default='meta-llama/Meta-Llama-3-8B-Instruct') # meta-llama/Meta-Llama-3-8B-Instruct  meta-llama/Llama-2-7b-chat-hf
    parser.add_argument('--target_model_name', '-tm', type=str, help='target_model_name', default='mistralai/Mistral-7B-Instruct-v0.1')
    # k is chosen from {5, 8, 10} and α is chosen from {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}.
    parser.add_argument('--k', '-k', type=str, help='k list', default='3,5')
    # reverse is: 1.6,1.5,1.4,1.3,1.2,1.15,1.1,1.05,1.00,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.5
    parser.add_argument('--alpha', '-alpha', type=float, default=0.5)
    # parser.add_argument('--alpha', '-alpha', type=str, help='trade-off alpha list', default='0.5,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.1,1.15,1.2,1.3,1.4,1.5,1.6')
    parser.add_argument('--option', '-o', type=str, help='either include or exclude', default='include') # 'include_switch'
    parser.add_argument('--split', '-split', type=str, help='run on dev or test', default='dev')
    parser.add_argument('--decoding_len', '-len', type=int, help='decoding length', default=128)
    parser.add_argument('--device_id', '-id', type=int, help='gpu id', default=0)
    parser.add_argument('--suffix', '-suffix', type=str, help='folder suffix', default='')
    parser.add_argument('--window_size', '-w', type=int, help='the window size', default=128)
    parser.add_argument('--seed', '-s', type=int, help='random seed', default=123)
    parser.add_argument('--debug', '-d', action='store_true', help='whether to debug')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite')
    parser.add_argument('--mode', type=str, default="vanilla_gen")
    parser.add_argument('--output_file', type=str, default="./text_output.json")
    parser.add_argument('--dataset', type=str, default="agnews")
    parser.add_argument('--input_path', type=str, default="/nas/home/qliu4174/PPL_gen/agnews/Which newspaper section would this article likely appear in?_vanilla_paraphrase.json")
    parser.add_argument('--output_mono_gen', type=str, default="/nas/home/qliu4174/PPL_gen/agnews/mono_gen/")
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default="ag_news")
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--start_idx', type=int, default=0)

    args = parser.parse_args()

    args.k = args.k.split(',')
    option = args.option
    if '.' in args.k[0]:
        args.k = list(map(float, args.k))
    else:
        args.k = list(map(int, args.k))

    set_seed(args.seed)

    if "llamakkk" in args.model_name:
        # insts_set = OOD_clean_instructions
        insts_set = long_insts
    else:
        insts_set = long_insts

    start = args.start_idx
    end = min(start + 13, len(insts_set[args.dataset_name]))

    input_prompts = insts_set[args.dataset_name][start:end]
    output_path = f"/nas02/qinliu/PPL_gen/rebuttle/{args.model_name.replace('/', '-')}/{args.dataset_name.replace('/', '-')}/"
    os.makedirs(output_path, exist_ok=True)
    output_file = f"{output_path}/results_data_first_{end}.json"

    # input_prompts = [" In this task, you're given a short article. Your job is to classify the article based on its category. Use the following classification labels, World, Sports, Business, Sci/Tech. Label the text \"World\" if it contains information related to world. Label the text \"Sports\" if it contains information related to sports. Label the text \"Business\" if it contains information related business. Label the text \"Sci/Tech\" if it contains science or technical related information."]
    # input_prompts = ["In this task, you are given a news article. Your task is to classify the article to one out of the four topics 'World', 'Sports', 'Business', 'Sci/Tech' if the article's main topic is relevant to the world, sports, business, and science/technology, correspondingly. If you are not sure about the topic, choose the closest option."]
    # output_file = "/nas02/qinliu/PPL_gen/sv002/ag_news_natural_insts/results_3.json"

    json_outputs = []

    # model = AutoModelForCausalLM.from_pretrained(args.model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = None
    # tokenizer = None
    split = {"ag_news": "test", "imdb": "test", "cola": "validation", "gimmaru/newspop": "train", "dair-ai/emotion": "test", "offensive": "validation",
             "sst2": "validation", "fancyzhx/dbpedia_14": "test"}
    content_name = {"ag_news": "text", "imdb": "text", "cola": "sentence", "gimmaru/newspop": "headline", "dair-ai/emotion": "text", "offensive": "text",
                    "sst2": "sentence", "fancyzhx/dbpedia_14": "content"}

    args.content_name = content_name[args.dataset_name]

    if args.dataset_name == "cola":
        args.task_name = "glue"
    elif args.dataset_name == "offensive":
        args.task_name = "tweet_eval"

    if args.task_name is None:
        ori_dataset = load_dataset(args.dataset_name, split=split[args.dataset_name])
    else:
        ori_dataset = load_dataset(args.task_name, args.dataset_name, split="validation")

    num_samples = min(1000, len(ori_dataset[args.content_name]))
    dataset = ori_dataset.shuffle(seed=123).select(range(num_samples))
    labels = LABEL_SET[args.dataset_name]

    # demonstrations = "Text: The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com) SPACE.com - TORONTO, Canada -- A second team of rocketeers competing for the #36;10 million Ansari X Prize, a contest for privately funded suborbital space flight, has officially announced the first launch date for its manned rocket.\n Output: Sci/Tech\n" \
    #                  "Text: Retailers Vie for Back-To-School Buyers (Reuters) Reuters - Apparel retailers are hoping their back-to-school fashions will make the grade among style-conscious teens and young adults this fall, but it could be a tough sell, with students and parents keeping a tighter hold on their wallets.\n Output: Business\n"

    # last_sent = demonstrations + "Choices: " + ", ".join(labels) + ". Answer with one word:"
    last_sent = "Choices: " + ", ".join(labels) + ". Answer with one word:"

    label_domain = []
    for label in labels:
        token_id = tokenizer.encode(label, return_tensors="pt")
        label_domain.append(token_id[0][1])

    mono_para_iter = []
    for prompt in tqdm(input_prompts):
        for iter in range(args.max_iter):
            if iter == 0:
                greedy_ppl, greedy_acc, ori_acc, ori_ppl = 0., 0., 0., 0.

                avg_ppl_ori = eval_avg_instance_ppl(prompt, dataset, model, tokenizer, args)

                greedy_para, greedy_ppl = greedy_gen(prompt, args)
                greedy_acc = eval_downstream(greedy_para, dataset, last_sent, label_domain, model, tokenizer, args)
                bert_score_g = eval_paraphrase(prompt, greedy_para)
                avg_ppl_greedy = eval_avg_instance_ppl(greedy_para, dataset, model, tokenizer, args)

                _, mono_para, iter, mono_ppl = run_PPL_greedy([prompt], args.model_name, 0, option, args.device_id, iter,
                                                     decoding_len=128, alpha=args.alpha)
                mono_para_iter.append(mono_para[0])
                mono_acc = eval_downstream(mono_para[0], dataset, last_sent, label_domain, model, tokenizer, args)
                bert_score_m = eval_paraphrase(prompt, mono_para[0])
                avg_ppl_mono = eval_avg_instance_ppl(mono_para[0], dataset, model, tokenizer, args)

                ori_acc = eval_downstream(prompt, dataset, last_sent, label_domain, model, tokenizer, args)
                ori_ppl = eval_PPL(prompt, args, model, tokenizer)

                json_outputs.append({"ori_prompt": prompt, "iter": iter, "mono_para": mono_para[0], "mono_ppl": mono_ppl,
                                 "bert_score_m": bert_score_m,
                                 "greedy_para": greedy_para, "greedy_ppl": greedy_ppl,
                                 "bert_score_g": bert_score_g,
                                 "ori_acc": ori_acc, "ori_ppl": ori_ppl, "greedy_acc": greedy_acc,
                                 "mono_acc": mono_acc, "avg_ppl_ori": avg_ppl_ori, "avg_ppl_greedy": avg_ppl_greedy,
                                     "avg_ppl_mono": avg_ppl_mono})

            else:
                _, mono_para, iter, mono_ppl = run_PPL_greedy([mono_para[0]], args.model_name, 0, option, args.device_id,
                                                              iter, decoding_len=128, alpha=args.alpha)
                if mono_para[0] in mono_para_iter:
                    mono_para_iter = []
                    break
                else:
                    mono_para_iter.append(mono_para[0])
                    mono_acc = eval_downstream(mono_para[0], dataset, last_sent, label_domain, model, tokenizer, args)
                    bert_score_m = eval_paraphrase(prompt, mono_para[0])
                    avg_ppl_mono = eval_avg_instance_ppl(mono_para[0], dataset, model, tokenizer, args)

                    json_outputs.append(
                        {"ori_prompt": prompt, "iter": iter, "mono_para": mono_para[0], "mono_ppl": mono_ppl,
                         "bert_score_m": bert_score_m,
                         "greedy_para": greedy_para, "greedy_ppl": greedy_ppl,
                         "bert_score_g": bert_score_g,
                         "ori_acc": ori_acc, "ori_ppl": ori_ppl, "greedy_acc": greedy_acc, "mono_acc": mono_acc, "avg_ppl_ori": avg_ppl_ori, "avg_ppl_greedy": avg_ppl_greedy,
                                     "avg_ppl_mono": avg_ppl_mono})
        mono_para_iter = []

    with open(output_file, 'a', encoding="utf-8") as file:
        json.dump(json_outputs, file, ensure_ascii=False, indent=4)


