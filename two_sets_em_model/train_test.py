# from __future__ import print_function
"""
add RL loss
"""

from data import *
from model import Model
from data import *
import json
from utils_c import *

model_name = 'erae_model'
time_str = get_time_str()
log_name = 'log\\train_log_' + time_str + model_name + '.log'
log_base = get_info_log(log_name, 'base')
time_str += model_name

def main_base():

    dataloader = DataLoader()
    model = Model(dataloader.params)
    model.model_prefix = './saved_erae_model/'
    model.model_path = model.model_prefix + 'model.ckpt'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    n_batch = len(dataloader.enc_inp) // args.batch_size

    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    model.initializer(sess)

    model.load_model(sess, log_base)

    log = None
    step = 0

    for epoch in range(args.stage_1_num_epochs):
        dataloader.update_word_dropout()
        log_base.info("Word Dropout")
        dataloader.shuffle()
        log_base.info("Data Shuffled")

        for i, (enc_inp, dec_inp, dec_out, labels, scores) in enumerate(dataloader.next_batch()):

            # increase step
            model.increase_step_session(sess)

            if i % args.stage_1_display_step == 0:
                log_base.info("------------")

            if step < args.stage_1_num_steps:

                log = model.train_vae_session(sess, enc_inp, dec_inp, dec_out, labels)
                print(log)
                step = log['step']
                if i % args.stage_1_display_step == 0:

                    log_base.info("Step %d | Train VAE | [%d/%d] | [%d/%d]" % (
                        log['step'], epoch+1, args.stage_1_num_epochs, i, n_batch))
                    log_base.info(" | nll_loss:%.1f" % (
                        log['nll_loss']))

            if step >= args.stage_1_num_steps:
                log = model.train_vae_embed_nll_session(sess, enc_inp, dec_inp, dec_out, labels)
                print(log)
                step = log['step']
                if i % args.stage_1_display_step == 0:

                    log_base.info("Step %d | Train VAE | [%d/%d] | [%d/%d]" % (
                        log['step'], epoch+1, args.stage_1_num_epochs, i, n_batch))
                    log_base.info(" | embed_nll_loss:%.1f" % (
                        log['embed_loss']))
            
            if i % args.stage_1_display_step == 0:
                log_base.info("------------")

            if i % (5 * args.stage_1_display_step) == 0 and log is not None:
                model.save_model(sess, log['step'])
                
            if i >= n_batch - 2:
                break

    save_path = model.saver.save(sess, model.model_path)
    log_base.info("Model saved in file: %s" % save_path)


def evaluate(sample_sent):
    config = tf.ConfigProto()
    
    sess = tf.Session(config=config)

    dataloader = DataLoader()
    model = Model(dataloader.params)
    model.model_prefix = './saved_erae_model/'
    model.model_path = model.model_prefix + 'model.ckpt'
    

    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    model.initializer(sess)

    model.load_model(sess, log_base)
    
    res = model.predict_dp_two_sets(sess, sample_sent, 0)[0]
    return res

if __name__ == '__main__':
    epsilons = [0, 10, 5, 3, 1, 0.9, 0.5, 0.1]
    sample_sent = "Markdown allows you to use backslash escapes to generate literal characters."
    if args.mode == 0:
        main_base()
    else:
        args.epsilon = epsilons[args.mode]
        log_base.info(json.dumps(args.__dict__, indent=4))
        print(evaluate(sample_sent))