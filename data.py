import os
import json
import re
import string
import numpy as np
from tqdm import tqdm
from sumeval.metrics.rouge import RougeCalculator
from sacrebleu import corpus_bleu


import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

class QAData(object):

    def __init__(self, logger, args, data_path, is_training):
        self.data_path = data_path
        if args.debug:
            self.data_path = data_path.replace("train", "dev")
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        if type(self.data)==dict:
            self.data = self.data["data"]
        if args.debug:
            self.data = self.data[:40]
        assert type(self.data)==list
        # assert all(["id" in d for d in self.data]), self.data[0].keys()
        # if type(self.data[0]["id"])==int:
        #     for i in range(len(self.data)):
        #         self.data[i]["id"] = str(self.data[i]["id"])
        #
        # self.index2id = {i:d["id"] for i, d in enumerate(self.data)}
        # self.id2index = {d["id"]:i for i, d in enumerate(self.data)}
        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args
        if "test" in self.data_path:
            self.data_type = "test"
        elif "dev" in self.data_path:
            self.data_type = "dev"
        elif "train" in self.data_path:
            self.data_type = "train"
        else:
            raise NotImplementedError()
        self.metric = "EM"
        self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None
        self.ref = None

    def __len__(self):
        return len(self.ref)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).lower()

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(".json", "-{}.json".format(postfix)))
        preprocessed_ref_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(".json", "-{}_ref.json".format(postfix)))
        if self.load and os.path.exists(preprocessed_path):
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                input_ids_snippet, attention_mask_snippet, input_ids_scenario, attention_mask_scenario, input_ids_question, attention_mask_question, decoder_input_ids, decoder_attention_mask = json.load(f)
            self.logger.info("Loading pre-tokenized ground truth from {}".format(preprocessed_ref_path))
            with open(preprocessed_ref_path, "r") as f:
                self.ref = json.load(f)
        else:
            print ("Start tokenizing...")

            snippets = []
            scenarioes = []
            questions = []
            answers = []
            
            for dialogue in self.data:
                history = dialogue["question"]
                for utterance in dialogue["history"]:
                    snippets.append(dialogue["snippet"])
                    scenarioes.append(dialogue["scenario"])
                    questions.append(history)
                    answers.append(utterance["follow_up_question"])
                    history += " " + utterance["follow_up_question"] + " " + utterance["follow_up_answer"]
                snippets.append(dialogue["snippet"])
                scenarioes.append(dialogue["scenario"])
                questions.append(history)
                answers.append(dialogue["answer"])
                
            if self.args.do_lowercase:
                snippets = [snippet.lower() for snippet in snippets]
                scenarioes = [scenario.lower() for scenario in scenarioes]
                questions = [question.lower() for question in questions]
                answers = [answer.lower() for answer in answers]
            if self.args.append_another_bos:
                snippets = ["<s> " + snippet for snippet in snippets]
                scenarioes = ["<s> " + scenario for scenario in scenarioes]
                questions = ["<s> "+question for question in questions]
                answers = ["<s> " +answer for answer in answers]

            snippet_input = tokenizer.batch_encode_plus(snippets,
                                                        pad_to_max_length=True,
                                                        max_length=self.args.max_input_length)
            scenario_input = tokenizer.batch_encode_plus(scenarioes,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_input_length)
            question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_input_length)
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length=True,
                                                       max_length=self.args.max_input_length)

            input_ids_snippet, attention_mask_snippet = snippet_input["input_ids"], snippet_input["attention_mask"]
            input_ids_scenario, attention_mask_scenario = scenario_input["input_ids"], scenario_input["attention_mask"]
            input_ids_question, attention_mask_question = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            if self.load:
                preprocessed_data = [input_ids_snippet, attention_mask_snippet,
                                     input_ids_scenario, attention_mask_scenario,
                                     input_ids_question, attention_mask_question,
                                     decoder_input_ids, decoder_attention_mask]
                with open(preprocessed_path, "w") as f:
                    json.dump(preprocessed_data, f)
                with open(preprocessed_ref_path, "w") as f:
                    json.dump(answers, f)
                    
            self.ref = answers
            
        self.dataset = MyQADataset(input_ids_snippet, attention_mask_snippet,
                                   input_ids_scenario, attention_mask_scenario,
                                   input_ids_question, attention_mask_question,
                                   decoder_input_ids, decoder_attention_mask,
                                   is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        ems = []
        rouge1s = []
        rouge2s = []
        rougels = []
        rouge = RougeCalculator(stopwords=True, lang="en")
        bleus = []
        
        for (prediction, ref) in zip(predictions, self.ref):
            ems.append(get_exact_match(prediction, ref))
            rouge_1 = rouge.rouge_n(
                summary=prediction,
                references=ref,
                n=1)
            rouge1s.append(rouge_1)
            rouge_2 = rouge.rouge_n(
                summary=prediction,
                references=ref,
                n=2)
            rouge2s.append(rouge_2)
            rouge_l = rouge.rouge_l(
                summary=prediction,
                references=ref)
            rougels.append(rouge_l)
            s = corpus_bleu("<s>" + prediction, ref).score
            bleus.append(s)
            
        ems_score = np.mean(ems)
        rouge1s_score = np.mean(rouge1s)
        rouge2s_score = np.mean(rouge2s)
        rougels_score = np.mean(rougels)
        bleus_score = np.mean(bleus)
        
        return ems_score, rouge1s_score, rouge2s_score, rougels_score, bleus_score

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        prediction_dict = {dp["id"]:prediction for dp, prediction in zip(self.data, predictions)}
        save_path = os.path.join(self.args.output_dir, "{}predictions.json".format(self.args.prefix))
        with open(save_path, "w") as f:
            json.dump(prediction_dict, f)
        self.logger.info("Saved prediction in {}".format(save_path))

def get_exact_match(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


class MyQADataset(Dataset):
    def __init__(self,
                 input_ids_snippet, attention_mask_snippet,
                 input_ids_scenario, attention_mask_scenario,
                 input_ids_question, attention_mask_question,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata_snippet=None, in_metadata_scenario=None,
                 in_metadata_question=None, out_metadata=None,
                 is_training=False):
        self.input_ids_snippet = torch.LongTensor(input_ids_snippet)
        self.attention_mask_snippet = torch.LongTensor(attention_mask_snippet)
        self.input_ids_scenario = torch.LongTensor(input_ids_scenario)
        self.attention_mask_scenario = torch.LongTensor(attention_mask_scenario)
        self.input_ids_question = torch.LongTensor(input_ids_question)
        self.attention_mask_question = torch.LongTensor(attention_mask_question)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        
        self.in_metadata_snippet = list(zip(range(len(input_ids_snippet)), range(1, 1+len(input_ids_snippet)))) \
            if in_metadata_snippet is None else in_metadata_snippet
        self.in_metadata_scenario = list(zip(range(len(input_ids_scenario)), range(1, 1 + len(input_ids_scenario)))) \
            if in_metadata_scenario is None else in_metadata_scenario
        self.in_metadata_question = list(zip(range(len(input_ids_question)), range(1, 1 + len(input_ids_question)))) \
            if in_metadata_question is None else in_metadata_question

        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids_snippet)==len(self.attention_mask_snippet)==self.in_metadata_snippet[-1][-1]
        assert len(self.input_ids_scenario) == len(self.attention_mask_scenario) == self.in_metadata_scenario[-1][-1]
        assert len(self.input_ids_question) == len(self.attention_mask_question) == self.in_metadata_question[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata_snippet)

    def __getitem__(self, idx):
        if not self.is_training:
            idx_snippet = self.in_metadata_snippet[idx][0]
            idx_scenario = self.in_metadata_scenario[idx][0]
            idx_question = self.in_metadata_question[idx][0]
            return self.input_ids_snippet[idx_snippet], self.attention_mask_snippet[idx_snippet],\
                   self.input_ids_scenario[idx_scenario], self.attention_mask_scenario[idx_scenario],\
                   self.input_ids_question[idx_question], self.attention_mask_question[idx_question]

        in_idx_snippet = np.random.choice(range(*self.in_metadata_snippet[idx]))
        in_idx_scenario = np.random.choice(range(*self.in_metadata_scenario[idx]))
        in_idx_question = np.random.choice(range(*self.in_metadata_question[idx]))

        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids_snippet[in_idx_snippet], self.attention_mask_snippet[in_idx_snippet], \
               self.input_ids_scenario[in_idx_scenario], self.attention_mask_scenario[in_idx_scenario], \
               self.input_ids_question[in_idx_question], self.attention_mask_snippet[in_idx_question],\
               self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]

class MyDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)


