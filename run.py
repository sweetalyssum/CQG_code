import os
import numpy as np
import torch


from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from data import QAData
from bart import MyBart

def run(args, logger):
    tokenizer = BartTokenizer.from_pretrained(args.BART)

    train_data = QAData(logger, args, args.train_file, True)
    dev_data = QAData(logger, args, args.predict_file, False)

    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    if args.do_train:
        if args.checkpoint is not None:
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key):value for key, value in state_dict.items()}
            model = MyBart.from_pretrained(args.BART,
                                           state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
        else:
            model = MyBart.from_pretrained(args.BART)
        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=100000)
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_predict:
        checkpoint = os.path.join(args.output_dir, 'best-model.pt')
        def convert_to_single_gpu(state_dict):
            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key
            return {_convert(key):value for key, value in state_dict.items()}
        model = MyBart.from_pretrained(args.BART,
                                       state_dict=convert_to_single_gpu(torch.load(checkpoint)))
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()
        ems_score, rouge1s_score, rouge2s_score, rougels_score, bleus_score = inference(model, dev_data, save_predictions=True)
        logger.info("EM on %s data: %.5f" % (dev_data.data_type, ems_score))
        logger.info("Rouge-1 on %s data: %.5f" % (dev_data.data_type, rouge1s_score))
        logger.info("Rouge-2 on %s data: %.5f" % (dev_data.data_type, rouge2s_score))
        logger.info("Rouge-L on %s data: %.5f" % (dev_data.data_type, rougels_score))
        logger.info("Bleu on %s data: %.5f" % (dev_data.data_type, bleus_score))

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in train_data.dataloader:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            model_output = model(input_ids_snippet=batch[0], attention_mask_snippet=batch[1],
                         input_ids_scenario=batch[2], attention_mask_scenario=batch[3],
                         input_ids_question=batch[4], attention_mask_question=batch[5],
                         decoder_input_ids=batch[6], decoder_attention_mask=batch[7],
                         is_training=True)
            loss = model_output["loss"]
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()

            if global_step % args.eval_period == 0:
                model.eval()
                curr_ems_score, curr_rouge1s_score, curr_rouge2s_score, curr_rougels_score, curr_bleus_score = inference(model if args.n_gpu==1 else model.module, dev_data, save_predictions=True)
                logger.info("Step %d Train loss %.5f %s %.5f %s %.5f %s %.5f %s %.5f %s %.5f on epoch=%d" % (
                        global_step,
                        np.mean(train_losses),
                        "EM",
                        curr_ems_score,
                        "Rouge-1",
                        curr_rouge1s_score,
                        "Rouge-2",
                        curr_rouge2s_score,
                        "Rouge-L",
                        curr_rougels_score,
                        "Bleu",
                        curr_bleus_score,
                        epoch))
                train_losses = []
                if best_accuracy < curr_bleus_score:
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                    logger.info("Saving model with best %s: %.5f -> %.5f on epoch=%d, global_step=%d" % \
                            ("Bleu", best_accuracy, curr_bleus_score, epoch, global_step))
                    best_accuracy = curr_bleus_score
                    wait_step = 0
                    stop_training = False
                # else:
                #     wait_step += 1
                #     if wait_step >= args.wait_step:
                #         stop_training = True
                #         break
                model.train()
        if stop_training:
            break

def inference(model, dev_data, save_predictions=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    for i, batch in enumerate(dev_data.dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        outputs = model.generate(input_ids_snippet=batch[0], attention_mask_snippet=batch[1],
                                 input_ids_scenario=batch[2], attention_mask_scenario=batch[3],
                                 input_ids_question=batch[4], attention_mask_question=batch[5],
                                 num_beams=dev_data.args.num_beams,
                                 max_length=dev_data.args.max_output_length,
                                 early_stopping=True)
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)
    if save_predictions:
        dev_data.save_predictions(predictions)
    ems_score, rouge1s_score, rouge2s_score, rougels_score, bleus_score = dev_data.evaluate(predictions)
    return ems_score, rouge1s_score, rouge2s_score, rougels_score, bleus_score







