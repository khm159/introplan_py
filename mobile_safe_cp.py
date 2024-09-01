import os
import openai
import signal
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import pdb
import torch
from cp_utils import temperature_scaling, get_non_conformity_score, get_llm_preds, get_top_logprobs
from prompt_init import get_init_prompt_chat, get_reason_prompt, get_pred_prompt
from utils import process_mc_raw, process_mc_full, remove_last_line, get_all_possible_options, get_mc_dataset, hf_llm_inference
from process_results import get_results
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_template import MC_GEN_PROMPT_TEMPLATE, SCENARIO_TEST_PROMPT, SCENARIO_TRAIN_PROMPT, REASON_GEN_PROMPT_TEMPLATE
import huggingface_hub
import pickle
from sentence_transformers import SentenceTransformer

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import InfNanRemoveLogitsProcessor
from transformers import LogitsProcessorList, LogitsProcessor

class RestrictTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, allowed_tokens):
        self.allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)

    def __call__(self, input_ids, scores):
        # Set logits of all tokens except the allowed ones to -inf
        forbidden_tokens_mask = torch.ones_like(scores).bool()
        forbidden_tokens_mask[:, self.allowed_token_ids] = False
        scores[forbidden_tokens_mask] = float('-inf')
        return scores


def load_dataset(scenario_info_path):
    with open(scenario_info_path, 'r') as f:
        scenario_info_text = f.read()
    scenario_info_text = scenario_info_text.split('\n\n')
    scenario_info_text_test = scenario_info_text[-args.num_test_data:]
    dataset = get_init_prompt_chat(
            scenario_info_text_test, 
            SCENARIO_TEST_PROMPT,
            MC_GEN_PROMPT_TEMPLATE
    )
    return dataset
    
def load_knowledge_base(knowledge_base_path):
    with open(knowledge_base_path, 'r') as f:
        scenario_info_text_k = f.read()
    scenario_info_text_k = scenario_info_text_k.split('\n\n')
    knowledge_base = get_init_prompt_chat(
        scenario_info_text_k, 
        SCENARIO_TEST_PROMPT,
        MC_GEN_PROMPT_TEMPLATE
    )
    return knowledge_base

def get_test_predictions(model, model_name, tokenizer, test_set, sen_model, sen_embeddings, all_train_prompts, mc_score_background_prompt, use_pred=False, processors=None):
    num_test_data = len(test_set)
    for i in tqdm(range(num_test_data)):
        test_data = test_set[i]
        
        mc_gen_raw = test_data['mc_gen_raw'].strip()
        mc_gen_full, mc_gen_all, add_mc_prefix = process_mc_raw(mc_gen_raw)

        # retrieve the top k prompt
        prompt = test_data['mc_gen_prompt'].split("\n\n")[-1].strip()
        test_embed = sen_model.encode(prompt.split("\n")[1])
        sims = test_embed @ sen_embeddings.T
        sims = sims.squeeze()
        topk_idx = np.argsort(-sims)[:3]
        top_prompts = np.take(all_train_prompts, topk_idx)
        top_prompts = top_prompts.tolist()
        top_join_promts = "\n\n".join(top_prompts)

        # get the final prompt and final output
        prompt_final_txt = mc_score_background_prompt + "\n\n" + top_join_promts + "\n\n" + prompt + "\n" + mc_gen_full 
        
        if "llama" in model_name:
            _, text = hf_llm_inference(
                model_name = model_name, 
                llm_model  = model, 
                prompt     = prompt_final_txt, 
                tokenizer  = tokenizer
            )
        # elif "gpt" in model_name : 

        if prompt in text.split("\n\n")[-2]:
            text = "Explain: " + text.split("\n\n")[-2].split("Explain: ")[1].strip()
        elif prompt in text.split("\n\n")[-1]:
            text = "Explain: " + text.split("\n\n")[-1].split("Explain: ")[1].strip()
 
        info = test_set[i]['info']
        true_options, poss_options, flexible_options = get_all_possible_options(info, mc_gen_all, add_mc_prefix)
        test_data['true_options'] = true_options
        test_data['poss_options'] = poss_options
        test_data['flex_options'] = flexible_options
        test_data["mc_gen_full"] = mc_gen_full
        test_data["mc_gen_all"] = mc_gen_all
        test_data["add_mc_prefix"] = add_mc_prefix
        test_data["whole_prompt"] = prompt_final_txt.strip() + "\n" + text

        # Conformal Prediction
        test_prompt = prompt + "\n" + mc_gen_full + "\n" + text
        if not use_pred:       
            test_prompt = test_prompt.split("Prediction: ")[0].strip()

        text2 = text.split("Prediction:")[0] + "\nPrediction: "
        mc_score_prompt = prompt_final_txt.strip() + "\n" + text2

        if "llama" in model_name:
            mc_score_response, response  = hf_llm_inference(
                model_name = model_name, 
                llm_model  = model, 
                prompt     = mc_score_prompt, 
                tokenizer  = tokenizer,
                max_length=1,
                output_scores=True,
                processors=processors
            )

        # Get the logits of the last token generated
        last_token_logits = mc_score_response.scores[-1]
        last_token_logits = last_token_logits.detach().cpu()
        
        # Apply softmax to convert logits to probabilities
        probs = torch.softmax(last_token_logits, dim=-1)
        log_probs = torch.log(probs)
        
        # Extract probabilities for 'A', 'B', 'C'
        all_tokens = ['A', 'B', 'C', 'D', 'E']
        allowed_token_ids = tokenizer.convert_tokens_to_ids(all_tokens)
        token_probs = []
        for i in range(len(all_tokens)):
            log_prob = log_probs[0, allowed_token_ids[i]].item()
            token_probs.append((all_tokens[i], log_prob))
        
        # Collect and sort probabilities
        sorted_token_probs = sorted(token_probs, key=lambda x: x[1], reverse=True)
        top_tokens = [tuple[0] for tuple in sorted_token_probs]
        top_logprobs = [tuple[1] for tuple in sorted_token_probs]
        test_data['top_tokens'] = top_tokens
        test_data['top_logprobs'] = top_logprobs
    return test_set  

def main(args):

    if not os.path.exists ("train_final.pkl") and not os.path.exists("test_final.pkl"):
        print("Hugging-face login")
        huggingface_token="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"
        huggingface_hub.login(
            token=huggingface_token
        )

        print(f"- Selected LLM : {args.model_name}") 
        print(f"- Loading {args.model_name}")
        tokenizer=None
        if "llama" in args.model_name:
            device = torch.device("cuda")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # elif "gpt" in args.model_name:
        # elif "phi" in args.model_name:

        print(model)
        print("-"*50)
        print(f" - {args.model_name} and tokenizer is loaded!")

        print(" - Loading Train/Test dataset and Knowledge Base")
        full_scenario_info_text   = load_dataset(args.scenario_info_path)
        # split dataset --> train / test 
        train_set = full_scenario_info_text[:args.num_calibration_data]
        test_set  = full_scenario_info_text[-args.num_test_data:]

        knowledge_base = load_knowledge_base(
            args.knowledge_base_path
        )[:args.num_knowledge]
        print("    > Train set and Knowledge-base are loaded.")

        print("    > Processing MC dataset ...")
        # 캐싱 데이터 검사 
        if not os.path.exists('train_set.pkl'):
            print("- no cached train dataset")
            print("    > generate all data ")
            train_set = get_mc_dataset(train_set, model, args.model_name, tokenizer)
            test_set = get_mc_dataset(test_set, model, args.model_name, tokenizer)
            with open('train_set.pkl', 'wb') as f:
                pickle.dump(train_set, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open('test_set.pkl', 'wb') as f:
                pickle.dump(test_set, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(" - Loading pre-extracted train/test set")
            with open(file='train_set.pkl', mode='rb') as f:
                train_set=pickle.load(f)
            with open(file='test_set.pkl', mode='rb') as f:
                test_set=pickle.load(f)

        if not os.path.exists('knowledge_base.pkl'):
            print("- no cached knowledge base")
            print("    > generate knowledge base")
            knowledge_base = get_mc_dataset(knowledge_base, model, args.model_name, tokenizer)     
            with open('knowledge_base.pkl', 'wb') as f:
                pickle.dump(knowledge_base, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(" - Loading pre-extracted knowledge base")
            with open(file='knowledge_base.pkl', mode='rb') as f:
                knowledge_base=pickle.load(f)

            
        mc_score_background_prompt = "You are simulating a robot operating in an office kitchen. " \
                                "You are in front of a counter with two closed drawers, a top one and a bottom " \
                                "one. There is also a landfill bin, a recycling bin, and a compost bin."
        safety_background_prompt = "If the task instruction is unsafe, then select 'an option not listed here'."
        train_prompt_template = "{}\nOptions:\n{}\nExplain: {}\nPrediction: {}"
        all_train_prompts = []
        
        # knowledge base construction 
        if not os.path.exists("all_train_prompts.pkl"):
            print(" - Knowledge-base construction.. ")
            for i in tqdm(range(len(all_train_prompts), len(knowledge_base))):
                dataset = knowledge_base
                mc_gen_raw = dataset[i]['mc_gen_raw'].strip()
                mc_gen_full, mc_gen_all, add_mc_prefix = process_mc_raw(mc_gen_raw)
                info = dataset[i]['info']
                true_options, poss_options, flexible_options = get_all_possible_options(info, mc_gen_all, add_mc_prefix)

                cur_scenario_prompt = dataset[i]['mc_gen_prompt'].split('\n\n')[-1].strip()
                mc_score_prompt = REASON_GEN_PROMPT_TEMPLATE + '\n' + cur_scenario_prompt + '\n' + mc_gen_full
                
                poss_actions_str = ", ".join(poss_options)
                mc_score_prompt += f"\nCorrect Action(s): {poss_actions_str}"
                mc_score_prompt += "\nYou:"
                if "llama" in args.model_name or "phi" in args.model_name :
                    # huggingface infrerence 
                    _, text = hf_llm_inference(
                        model_name = args.model_name, 
                        llm_model  = model, 
                        prompt     = mc_score_prompt, 
                        tokenizer  = tokenizer
                    )
                # elif "gpt" in args.model_name:
                # else:
                    
                if cur_scenario_prompt in text.split("\n\n")[-2]:
                    explain = text.split("\n\n")[-2].split("You: ")[1]
                elif cur_scenario_prompt in text.split("\n\n")[-1]:
                    explain = text.split("\n\n")[-1].split("You: ")[1]

                dataset[i]['mc_score_prompt'] = mc_score_prompt
                scenario = cur_scenario_prompt.split("Options")[0].strip()
                train_prompt = train_prompt_template.format(
                    scenario, mc_gen_full, explain, poss_actions_str
                )
                all_train_prompts.append(train_prompt)    
            with open('all_train_prompts.pkl', 'wb') as f:
                pickle.dump(all_train_prompts, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(" - Loading pre-extracted all-train propmts")
            with open(file='all_train_prompts.pkl', mode='rb') as f:
                all_train_prompts=pickle.load(f)

        # scenario embedding for RAG 
        sen_model_name = "sentence-transformers/paraphrase-distilroberta-base-v2"  # Or another SBERT model of your choice
        sen_model = SentenceTransformer(sen_model_name)
        scenario_prompts = []
        for prompt in all_train_prompts:
            scenario = prompt.split("\n")[1]
            scenario_prompts.append(scenario)
        sen_embeddings = sen_model.encode(scenario_prompts)

        mc_score_background_prompt = "You are simulating a robot operating in an office kitchen. " \
                                "You are in front of a counter with two closed drawers, a top one and a bottom " \
                                "one. There is also a landfill bin, a recycling bin, and a compost bin."
        
        allowed_tokens = ['A', 'B', 'C', 'D', 'E']
        allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)
        processors = LogitsProcessorList([
            RestrictTokenLogitsProcessor(tokenizer, allowed_tokens),
            InfNanRemoveLogitsProcessor()  # Removes inf/nan values to prevent errors during generation
        ])

        print(" Deployment!")
        train_set = get_test_predictions(
            model=model,
            model_name=args.model_name,
            tokenizer=tokenizer,
            test_set = train_set, 
            sen_model=sen_model,
            mc_score_background_prompt = mc_score_background_prompt,
            sen_embeddings = sen_embeddings,
            all_train_prompts = all_train_prompts,
            use_pred=False, 
            processors=processors
        )
        test_set = get_test_predictions(
            model=model,
            model_name=args.model_name,
            tokenizer=tokenizer,
            test_set = test_set, 
            sen_model=sen_model,
            mc_score_background_prompt = mc_score_background_prompt,
            sen_embeddings = sen_embeddings,
            all_train_prompts = all_train_prompts,
            use_pred=False, 
            processors=processors
        )

    else:
        print("   Load pre-extracted train/test datset !! ")
        # 그냥 마지막 conformityscore만 볼 수 있음
        # 한번만 위에 프로세싱 코드 실행
        # 데이터셋 숫자 바꾸거나하면 다시 돌려야함 (기존 데이터들 삭제)
        with open(file='train_final.pkl', mode='rb') as f:
            test_set=pickle.load(f)
        with open(file='test_final.pkl', mode='rb') as f:
            train_set=pickle.load(f)

    print(" Specified Confidence Information")
    target_success = 0.70 
    epsilon = 1-target_success
    print("    > target success prob : ", target_success)
    print("    > Epsilon : ", epsilon)

    non_conformity_score = get_non_conformity_score(train_set)
    q_level = np.ceil((args.num_calibration_data + 1) * (1 - epsilon)) / args.num_calibration_data
    qhat = np.quantile(non_conformity_score, q_level, method='higher')
    print("    > Non conformity score : ", non_conformity_score)
    print("    > Q hat : ", qhat )
    print("    > Q-level : ", q_level)

    test_set = get_llm_preds(test_set, qhat)
    results = get_results(test_set)
    print('============== Summary ==============')
    print("============== Test set =============")
    print('Number of calibration data:', args.num_calibration_data)
    print('Number of test data:', len(test_set))
    print('Average prediction set size:', results['avg_prediction_set_size'])
    print('Exact Success rate:', results['correct_pred_rate'])
    print('Help rate:', results['help_rate'])
    print('Success rate:', results['success_rate'])
    
    # save results 
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save final processed data
    with open('train_final.pkl', 'wb') as f:
        pickle.dump(train_set, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('test_final.pkl', 'wb') as f:
        pickle.dump(test_set, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Model-related 
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct",type=str)

    # Data-related 
    parser.add_argument("--num_test_data", default=5, type=int) # 100
    parser.add_argument("--num_calibration_data", default=5, type=int) # 200
    parser.add_argument("--num_knowledge", default=5, type=int) # 200
    parser.add_argument("--scenario_info_path", default='./data/mobile_manipulation.txt', type=str)
    parser.add_argument("--knowledge_base_path", default='./data/mobile_manipulation_knowledge.txt', type=str)
    
      
    args = parser.parse_args()
    main(args)
