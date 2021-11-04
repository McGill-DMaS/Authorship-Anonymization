import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_sample_size', type=int, default=5)
parser.add_argument('--vocab_size', type=int, default=20000)
parser.add_argument('--num_class', type=int, default=2)
parser.add_argument('--max_len', type=int, default=50)
parser.add_argument('--word_dropout_rate', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--embedding_dim', type=int, default=768)

parser.add_argument('--rnn_size', type=int, default=512)
parser.add_argument('--beam_width', type=int, default=15)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--latent_size', type=int, default=256)
parser.add_argument('--kl_anneal_max', type=float, default=1.0)
parser.add_argument('--kl_anneal_bias', type=int, default=6000)
parser.add_argument('--temperature_anneal_max', type=float, default=1.0)
parser.add_argument('--temperature_anneal_bias', type=int, default=6000)
parser.add_argument('--temperature_start_step', type=int, default=19000)

parser.add_argument('--use_bert', type=bool, default=True)

parser.add_argument('--lambda_n', type=float, default=1)
parser.add_argument('--lambda_embed', type=float, default=10)

parser.add_argument('--max_nll', type=float, default=100000)

parser.add_argument('--stage_1_num_epochs', type=int, default=50)
parser.add_argument('--stage_1_num_steps', type=int, default=15000)
parser.add_argument('--stage_1_display_step', type=int, default=50)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_layer', type=int, default=1)

parser.add_argument('--use_my_dict', type=bool, default=True)

parser.add_argument('--data_path', type=str, default="dataset\yelp\yelp.txt")
parser.add_argument('--id_path', type=str, default="dataset\yelp\id2w.txt")
parser.add_argument('--dict_path', type=str, default="my_dict.json")
parser.add_argument('--model_path', type=str, default="./saved_erae_model/")

parser.add_argument('--epsilon', type=float, default=5)
parser.add_argument('--delta', type=float, default=0.5)
parser.add_argument('--mode', type=int, default=0)

args = parser.parse_args()




