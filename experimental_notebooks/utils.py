# File containing useful functions
import torch
import numpy as np
from tqdm import tqdm

def get_encoder_hidden_states(model, tokenizer, input_text, token_pos, layer=-1):
    """
    Given an encoder model and some text, gets the encoder hidden states (in a given layer, by default the last) 
    on that input text (where the full text is given to the encoder).

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize
    encoder_text_ids = tokenizer(input_text, truncation=True, return_tensors="pt").input_ids.to(model.device)

    # forward pass
    with torch.no_grad():
        output = model(encoder_text_ids, output_hidden_states=True)

    # get the appropriate hidden states
    hs_tuple = output["hidden_states"]
    
    hs = hs_tuple[layer][0, token_pos].detach().cpu().numpy()

    return hs

def get_encoder_decoder_hidden_states(model, tokenizer, input_text, token_pos, layer):
    """
    Given an encoder-decoder model and some text, gets the encoder hidden states (in a given layer, by default the last) 
    on that input text (where the full text is given to the encoder).

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize
    encoder_text_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    decoder_text_ids = tokenizer("", return_tensors="pt").input_ids.to(model.device)

    # forward pass
    with torch.no_grad():
        output = model(encoder_text_ids, decoder_input_ids=decoder_text_ids, output_hidden_states=True)

    # get the appropriate hidden states
    hs_tuple = output["encoder_hidden_states"]
    hs = hs_tuple[layer][0, token_pos].detach().cpu().numpy()

    return hs

def get_decoder_hidden_states(model, tokenizer, input_text, token_pos, layer):
    """
    Given a decoder model and some text, gets the hidden states (in a given layer, by default the last) on that input text

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize (adding the EOS token this time)
    input_ids = tokenizer(input_text + tokenizer.eos_token, return_tensors="pt").input_ids.to(model.device)

    # forward pass
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)

    # probe the specified layer and token position
    hs_tuple = output["hidden_states"]
    hs = hs_tuple[layer][0, token_pos].detach().cpu().numpy()

    return hs

def get_hidden_states(model, tokenizer, input_text, token_pos, layer, model_type="encoder"):
    fn = {"encoder": get_encoder_hidden_states, "encoder_decoder": get_encoder_decoder_hidden_states,
          "decoder": get_decoder_hidden_states}[model_type]

    return fn(model, tokenizer, input_text, token_pos, layer)

# Functions for formating contrasting pairs
def format_imdb_1(text, label):
    """
    Given an imdb example ("text") and corresponding label (0 for negative, or 1 for positive), 
    returns a zero-shot prompt for that example (which includes that label as the answer).
    
    (This is just one example of a simple, manually created prompt.)
    """
    return "The following movie review expresses a " + ["negative", "positive"][label] + " sentiment:\n" + text

def format_imdb_2(text, label):
    """
    Given an imdb example ("text") and corresponding label (0 for negative, or 1 for positive), 
    returns an alternative zero-shot prompt for that example (which includes that label as the answer).
    
    (This is just one example of a simple, manually created prompt.)
    """
    return "Is the sentiment of " + text + " negative or positive? " + "Answer: " + ["negative", "positive"][label] 

def format_imdb_3(text, label):
    """
    Given an imdb example ("text") and corresponding label (0 for negative, or 1 for positive), 
    returns a zero-shot prompt for that example (which includes that label as the answer).
    
    (This is just one example of a simple, manually created prompt.)
    """
    return "Consider the sentiment of the following movie review:\n" + text + "\nDoes the above movie review express a " + ["negative", "positive"][label] + " sentiment? " + "Answer: " + "Yes"

def format_imdb(version, text, label):
    prompt_form = {"imdb_1": format_imdb_1, "imdb_2": format_imdb_2, "imdb_3": format_imdb_3}[version]
    return prompt_form(text, label)

def format_amazon(text, label):
    """
    Given an Amazon review example ("text") and corresponding label (0 for negative, or 1 for positive), 
    returns a zero-shot prompt for that example (which includes that label as the answer).
    
    (This is just one example of a simple, manually created prompt.)
    """
    return "The following review expresses a " + ["negative", "positive"][label] + " sentiment:\n" + text


# def get_hidden_states_many_examples(model, tokenizer, data, dataset_name, model_type, layer = -1, n=100):
#     """
#     Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples by probing the specified layer.
#     Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
#     with the ground truth labels
    
#     This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
#     """
#     # setup
#     model.eval()
#     all_neg_hs, all_pos_hs, all_gt_labels = [], [], []

#     # loop
#     for _ in tqdm(range(n)):
#         # for simplicity, sample a random example until we find one that's a reasonable length
#         # (most examples should be a reasonable length, so this is just to make sure)
#         while True:
#             idx = np.random.randint(len(data))
#             text, true_label = "hello", 0 
#             if dataset_name == "imdb":
#                text, true_label = data[idx]["text"], data[idx]["label"]
#             else:
#                text, true_label = data[idx]["content"], data[idx]["label"]
#             # the actual formatted input will be longer, so include a bit of a marign
#             if len(tokenizer(text)) < 400:  
#                 break
                
#         # get hidden states
#         neg_hs = get_hidden_states(model, tokenizer, format_imdb(text, 0), layer = layer, model_type=model_type)
#         pos_hs = get_hidden_states(model, tokenizer, format_imdb(text, 1), layer = layer, model_type=model_type)

#         # collect
#         all_neg_hs.append(neg_hs)
#         all_pos_hs.append(pos_hs)
#         all_gt_labels.append(true_label)

#     all_neg_hs = np.stack(all_neg_hs)
#     all_pos_hs = np.stack(all_pos_hs)
#     all_gt_labels = np.stack(all_gt_labels)

#     return all_neg_hs, all_pos_hs, all_gt_labels

# def get_hidden_states_aggregate(model, tokenizer, data, dataset_name, model_type, layers, token_pos = -1, num_samples = 100):
    ''' 
    Return all_neg_hs := a list of different hidden state representations of the complements
           all_pos_hs := a list of different hidden state representations of the events
           all_gt_labels := a list of true labels of the true labels
    '''
    model.eval()
    all_neg_hs, all_pos_hs, all_gt_labels = [], [], []

    for _ in tqdm(range(num_samples)):
        # for simplicity, sample a random example until we find one that's a reasonable length
        # (most examples should be a reasonable length, so this is just to make sure)
        idx = np.random.randint(len(data))
        text, true_label = "hello", 0 
        if dataset_name == "imdb":
            text, true_label = data[idx]["text"], data[idx]["label"]
        else:
            text, true_label = data[idx]["content"], data[idx]["label"]
                
        # get hidden states from specified layers
        neg_hs_different_states = []
        pos_hs_different_states = []

        for i in layers:
            neg_hs = get_hidden_states(model, tokenizer, format_imdb(text, 0), layer = i, model_type=model_type)
            pos_hs = get_hidden_states(model, tokenizer, format_imdb(text, 1), layer = i, model_type=model_type)
            neg_hs_different_states.append(neg_hs)
            pos_hs_different_states.append(pos_hs)
        
        all_neg_hs.append(neg_hs_different_states)
        all_pos_hs.append(pos_hs_different_states)
        all_gt_labels.append(true_label)
    
    return all_neg_hs, all_pos_hs, all_gt_labels


