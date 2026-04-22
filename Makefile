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
corpus_ssl_dir?=corpus_ssl
start_epoch?=1
gpus?=1

epoch?=30
avg?=10

##############################################################
download_dir?=data/download
data_dir?=data
vocab_size?=500
lang_dir=$(data_dir)/lang_bpe_${vocab_size}
lm_params?=
max_duration?=1000
seconds_train_kmeans?=360000
workers?=4
ssl_parts?=10

$(download_dir) $(data_dir)/manifests $(data_dir)/fbank \
	$(download_dir)/common_voice $(lang_dir) $(exp_dir) \
	$(data_dir)/kmeans $(data_dir)/tasks \
	$(corpus_ssl_dir):
	mkdir -p $@
info:
	@echo "corpus_dir: $(corpus_dir)"
	@echo "icefall_dir: $(icefall_dir)"
	@echo "vocab_size: $(vocab_size)"
	@echo "gpus: $(gpus)"
	@echo "train_params: $(train_params)"
	@echo "seconds_train_kmeans: $(seconds_train_kmeans)"
	@echo "max_duration: $(max_duration)"
	@echo "epoch: $(epoch)"
	@echo "avg: $(avg)"
	@echo "workers: $(workers)"
	@echo "ssl_parts: $(ssl_parts)"	
	@echo "ssl_feat_files: $(ssl_feat_files)"	
		
.PHONY: info	

install_deps:
	pip install joblib scikit-learn einops botocore==1.42.59 boto3
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
############################################################
### SSL DATA
############################################################
s3_from?=0
s3_to?=5
load/s3/%: | $(corpus_ssl_dir)
	$(python_cmd) ./SSL/local/s3_dwn.py --s3-bucket audio-corpus/$* --dest-dir $(corpus_ssl_dir) \
		--s3-from $(s3_from) --s3-to $(s3_to) --workers $(workers) 

# one tar is 5 hours of audio, so 5 tars is 25 hours, which is a good chunk to process at once
$(corpus_ssl_dir)/.loaded.ssl: | $(corpus_ssl_dir)
	make load/s3/crawl s3_from=0 s3_to=4 
# crawl-augmented 15h each
	make load/s3/crawl-augmented s3_from=20 s3_to=21 
	make load/s3/08kHz s3_from=0 s3_to=4 
	make load/s3/16kHz s3_from=0 s3_to=4 
	make load/s3/liepa3 s3_from=0 s3_to=4 
	make load/s3/voxlingua s3_from=0 s3_to=4 
# a lot of non lithuanian data, so skip
#	make load/s3/voxpopuli s3_from=0 s3_to=4 
	
	touch $@
load/ssl: $(corpus_ssl_dir)/.loaded.ssl
.PHONY: load/ssl
$(data_dir)/manifests/cuts_pretrain.jsonl.gz: $(corpus_ssl_dir)/.loaded.ssl | $(data_dir)/manifests 
	$(python_cmd) ./SSL/local/prepare_ssl_corpus.py --corpus-dir $(corpus_ssl_dir) --output-file $@ --workers $(workers)

shards := $(shell seq -f "%03g" 0 $$(($(ssl_parts)-1)))
ssl_cuts_files := $(foreach r,$(shards),$(data_dir)/manifests/cuts_pretrain_$(r).jsonl.gz)	
ssl_feat_files := $(foreach r,$(shards),$(data_dir)/fbank/cuts_pretrain_$(r).jsonl.gz)	

$(ssl_cuts_files): $(data_dir)/manifests/cuts_pretrain.jsonl.gz
	$(python_cmd) ./SSL/local/split_cuts.py --input $< --output-dir $(data_dir)/manifests --split-into $(ssl_parts)
$(data_dir)/fbank/%: $(data_dir)/manifests/% | $(data_dir)/fbank
	$(python_ssl_cmd) ./SSL/local/compute_fbank.py --input $^ --output $@ 

prepare/ssl: $(ssl_feat_files)
.PHONY: prepare/ssl
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
train: $(data_dir)/fbank/cuts_train.jsonl.gz | $(exp_dir)
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
    --use-averaged-model 1 \
    --cuts-name $*  
decode/test: _decode/test
.PHONY: train decode/test
##############################################################
# Learn Kmeans trained initial models
##############################################################
$(data_dir)/kmeans/kmeans.pt: $(data_dir)/fbank/cuts_train_100h.jsonl.gz | $(data_dir)/kmeans
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
    	--use-averaged-model 1 
learn/kmeans: $(data_dir)/kmeans/kmeans.pt
.PHONY: learn/kmeans
##############################################################
# Extract labels
##############################################################
$(data_dir)/tasks/extract.lists: | $(data_dir)/tasks
	$(python_cmd) ./SSL/local/make_task_list.py --template $(data_dir)/fbank/cuts_pretrain_{}.jsonl.gz --count $(ssl_parts) --output $@

extract/labels: $(data_dir)/kmeans/kmeans.pt $(data_dir)/tasks/extract.lists
	$(python_ssl_cmd) ./SSL/zipformer_fbank/extract_kmeans_scripts/extract_kmeans.py \
		--model-path $(data_dir)/kmeans/kmeans.pt \
    	--pretrained-dir $(exp_dir) \
		$(decode_params) \
    	--epoch $(epoch) \
    	--avg $(avg) \
    	--max-duration $(max_duration) \
    	--checkpoint-type ASR \
    	--use-averaged-model 1 \
		--task-list $(data_dir)/tasks/extract.lists
.PHONY: extract/labels
##############################################################
.EXPORT_ALL_VARIABLES:
