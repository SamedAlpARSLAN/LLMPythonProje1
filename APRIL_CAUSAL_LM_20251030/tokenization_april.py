from transformers import PreTrainedTokenizerFast

class AprilTokenizerHF(PreTrainedTokenizerFast):
    vocab_files_names = {}
    model_input_names = ["input_ids","attention_mask"]

    def __init__(self, *args, **kwargs):
        # HF tokenizer.json varsa otomatik okunur
        super().__init__(*args, **kwargs)
