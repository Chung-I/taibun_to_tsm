local model_name = "bert-base-chinese";
local max_length = 512;
local bert_dim = 768;
local bopomofo_dim = 128;
local encoder_dim = bert_dim + bopomofo_dim ;
local decoder_dim = 128;
local data_base_url = "https://storage.googleapis.com/allennlp-public-data/cnndm-combined-data-2020.07.13.tar.gz";
local train_data = "data/train.txt";
local dev_data = "data/dev.txt";
local test_data = "data/test.txt";

{
    "train_data_path": train_data,
    "validation_data_path": dev_data,
    "test_data_path": test_data,
    "dataset_reader": {
        "type": "sequence_tagging",
        "word_tag_delimiter": "|",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
                "max_length": max_length,
            },
            "bopomofo_tokens": {
                "type": "bpmf",
                "bos_token": "[CLS]",
                "eos_token": "[SEP]",
            },
        },
    },
    "model": {
        "type": "simple_tagger",
        "text_field_embedder" : {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": model_name,
                    "max_length": max_length,
                    "last_layer_only": false,
                },
                "bopomofo_tokens": {
                    "type": "character_encoding",
                    "embedding": {
                        "num_embeddings": 320,
                        // Same as the Transformer ELMo in Calypso. Matt reports that
                        // this matches the original LSTM ELMo as well.
                        "embedding_dim": 16
                    },
                    "encoder": {
                        "type": "cnn-highway",
                        "activation": "relu",
                        "embedding_dim": 16,
                        "filters": [
                            [1, 32],
                            [2, 32],
                            [3, 64],
                            [4, 128]],
                        "num_highway": 2,
                        "projection_dim": bopomofo_dim,
                        "projection_location": "after_highway",
                        "do_layer_norm": true
                    }
                }
            }
        },
        // "encoder": {
        //     "type": "pass_through",
        //     "input_dim": encoder_dim
        // },
        "encoder": {
            "type": "feedforward",
            "feedforward": {
                "input_dim": encoder_dim,
                "hidden_dims": decoder_dim,
                "num_layers": 1,
                "activations": "linear",
            }
        },
    },
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "trainer": {
        "num_epochs": 100,
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "adamw",
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "parameter_groups": [
                [[".*transformer.*"], {"lr": 1e-5}]
            ]
        },
        "patience": 10,
        "learning_rate_scheduler": {
          "type": "reduce_on_plateau",
          "factor": 0.5,
          "mode": "max",
          "patience": 5
        },
        "grad_norm": 5.0,
    }
}
