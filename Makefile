cfg?=Makefile.options
-include $(cfg)

LOGLEVEL?=INFO

exp_dir?=data/exp/v01

icefall_dir?=./../icefall

python_cmd=PYTHONPATH=$(icefall_dir) LOG_LEVEL=$(LOGLEVEL) python3
python_ssl_cmd=PYTHONPATH=./SSL/zipformer_fbank:$(icefall_dir) LOG_LEVEL=$(LOGLEVEL) python3
decode_limit?=0
limit_count?=100000000000000

corpus_dir?=
corpus_ln_dir?=corpus
start_epoch?=1
gpus?=1

epoch?=30
avg?=15

##############################################################
download_dir?=data/download
data_dir?=data
vocab_size?=500
lang_dir=$(data_dir)/lang_bpe_${vocab_size}
lm_params?=
max_duration?=1000
seconds_train_kmeans?=360000

$(download_dir) $(data_dir)/manifests $(data_dir)/fbank \
	$(download_dir)/common_voice $(lang_dir) $(exp_dir) \
	$(data_dir)/kmeans:
	mkdir -p $@
info:
	@echo "corpus_dir: $(corpus_dir)"
	@echo "icefall_dir: $(icefall_dir)"
	@echo "vocab_size: $(vocab_size)"
	@echo "gpus: $(gpus)"
	@echo "train_params: $(train_params)"
	@echo "seconds_train_kmeans: $(seconds_train_kmeans)"
			
.PHONY: info	

install_deps:
	pip install joblib scikit-learn
##############################################################
##############################################################
# Liepa 3 prepare											 #
##############################################################
$(corpus_ln_dir): | $(corpus_dir)
	ln -s $(corpus_dir) $@

$(data_dir)/manifests/cuts_train.jsonl.gz $(data_dir)/manifests/cuts_dev.jsonl.gz $(data_dir)/manifests/cuts_test.jsonl.gz: | $(data_dir)/manifests $(corpus_ln_dir)
	$(python_cmd) $(icefall_dir)/egs/liepa3/ASR/local/prepare_corpus.py --corpus-dir $(corpus_ln_dir) --transcript-file $(corpus_ln_dir)/verified_utterances_20260121.csv \
          --output-dir $(data_dir)/manifests --limit-count $(limit_count)
    # Split train/dev/test (90/5/5)
	$(python_cmd) $(icefall_dir)/egs/liepa3/ASR/local/split.py --manifest-dir $(data_dir)/manifests

$(data_dir)/fbank/cuts_train.jsonl.gz $(data_dir)/fbank/cuts_dev.jsonl.gz $(data_dir)/fbank/cuts_test.jsonl.gz: $(data_dir)/manifests/cuts_train.jsonl.gz $(data_dir)/manifests/cuts_dev.jsonl.gz $(data_dir)/manifests/cuts_test.jsonl.gz | $(data_dir)/fbank
	$(python_cmd) $(icefall_dir)/egs/liepa3/ASR/local/compute_fbank_liepa3.py --input-dir $(data_dir)/manifests --output-dir $(data_dir)/fbank

$(data_dir)/fbank/.%.validated: $(data_dir)/fbank/cuts_%.jsonl.gz
	$(python_cmd) $(icefall_dir)/egs/liepa3/ASR/local/validate_manifest.py $<
	touch $@

$(lang_dir)/transcript_words.txt: $(data_dir)/manifests/cuts_train.jsonl.gz | $(lang_dir)
	$(python_cmd) $(icefall_dir)/egs/liepa3/ASR/local/prepare_texts.py --input-file $< --output-file $@

$(lang_dir)/bpe.model: $(lang_dir)/transcript_words.txt
	$(python_cmd) $(icefall_dir)/egs/liepa3/ASR/local/train_bpe_model.py --lang-dir $(lang_dir) \
		--vocab-size $(vocab_size) --transcript $<

$(data_dir)/fbank/cuts_train_100h.jsonl.gz: $(data_dir)/fbank/cuts_train.jsonl.gz
	$(python_cmd) ./ASR/local/take_cuts.py --input $< --output $@ --secs $(seconds_train_kmeans)

prepare/liepa3: $(data_dir)/fbank/.train.validated $(data_dir)/fbank/.dev.validated $(data_dir)/fbank/.test.validated $(lang_dir)/bpe.model \
	$(data_dir)/fbank/cuts_train_100h.jsonl.gz
.PHONY: prepare/liepa3
##############################################################
# Train Initial LIEPA3 model                                 #
##############################################################
train_params?=--use-fp16 1 --train-cuts 4000h --max-duration $(max_duration) --enable-musan 0 --enable-spec-aug 1 --seed 1332 --master-port 12356 \
	--bpe-model $(lang_dir)/bpe.model \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 768,1536,2048,3072,2048,1536 \
    --encoder-dim 256,512,768,1024,768,512 \
    --encoder-unmasked-dim 256,256,256,320,256,256 \
    --base-lr 0.045
train: | $(exp_dir)
	$(python_cmd) ./ASR/zipformer/train.py --world-size $(gpus) \
		--num-epochs $(epoch) --start-epoch $(start_epoch) \
		--bpe-model $(lang_dir)/bpe.model --manifest-dir $(data_dir)/fbank \
		--exp-dir $(exp_dir) \
		$(train_params) 

decode_params?=--bpe-model $(lang_dir)/bpe.model \
	--num-encoder-layers 2,2,4,5,4,2 \
	--feedforward-dim 768,1536,2048,3072,2048,1536 \
	--encoder-dim 256,512,768,1024,768,512 \
	--encoder-unmasked-dim 256,256,256,320,256,256
_decode/%: 
	$(python_cmd) ./ASR/zipformer/decode.py \
    --epoch $(epoch) \
    --avg $(avg) \
    --exp-dir $(exp_dir) \
    --max-duration $(max_duration) \
	$(decode_params) \
    --decoding-method greedy_search \
    --manifest-dir $(data_dir)/fbank \
    --use-averaged-model 0 \
    --cuts-name $*  
decode/test: _decode/test
.PHONY: train decode/test
##############################################################
# Learn Kmeans trained initial models
##############################################################
$(data_dir)/kmeans/kmeans.pt: $(data_dir)/fbank/cuts_train_100h.jsonl.gz | $(data_dir)/kmeans
# 	 -m zipformer_fbank.extract_kmeans_scripts.learn_kmeans
	$(python_ssl_cmd) ./SSL/zipformer_fbank/extract_kmeans_scripts/learn_kmeans.py \
		--km-path $(data_dir)/kmeans/kmeans.pt \
    	--n-clusters 500 \
    	--max-iter 100 \
    	--files $(data_dir)/fbank/cuts_train_100h.jsonl.gz \
    	--do-training \
    	--pretrained-dir $(exp_dir) \
    	--epoch $(epoch) \
		$(decode_params) \
    	--avg $(avg) \
    	--max-duration $(max_duration) \
    	--checkpoint-type ASR \
    	--use-averaged-model 0 \
    	--bpe-model $(lang_dir)/bpe.model
learn/kmeans: $(data_dir)/kmeans/kmeans.pt
.PHONY: learn/kmeans
##############################################################
.EXPORT_ALL_VARIABLES:
