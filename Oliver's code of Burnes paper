import sys
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM

MODELS = {
  "llama-7b": "decapoda-research/llama-7b-hf",
  "llama-13b": "decapoda-research/llama-13b-hf",
  "llama-30b": "decapoda-research/llama-30b-hf",
  "llama-65b": "decapoda-research/llama-65b-hf"
}

MODEL_TAG = "llama-7b"
MODEL_NAME = MODELS[MODEL_TAG]

tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)

model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    # Further quantization (requries BitsAndBytes, experimental). Keep dtype=float16 with this
    # load_in_8bit=LOAD_8BIT,
    torch_dtype=torch.float16,
    # `device_map` maps layers and the lm head to devices they live on. `auto` works, `sequential`
    # and `balance_low0` should work but don't
    device_map="auto",
    # If the first GPU ram fills up and then you get a CUDA out of memory error, you may need to
    # manually specify the max memory per card. I don't know why accelerate / huggingface can't
    # always infer this. For 4x A6000:
    # max_memory = {0: "44gib", 1: "44gib", 2: "44gib", 3: "44gib"}
)

print("Loaded {}!".format(MODEL_TAG))
print(model)


from transformers import GenerationConfig

# Simple generation using huggingface's default interface. This will probably
# produce output that's pretty bad since LLaMA is a foundation model and hasn't
# been tuned on any downstream objective.

# Also doesn't use any smart heuristic for stopping so it will just keep generating
# until it hits the `max_new_tokens`
def generate(
    text,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output

# For messing around with generation

prefix = input()
completion = generate(prefix,temperature=0.8)
print(completion)

from tqdm import tqdm
import numpy as np

# Extract LLaMA hidden states from some sequence of tokens `input_ids`.
# Returns activations from the layer numbered `layer` (or -1 for last layer)
def llama_hs_from_tokens(model, input_ids, layer=-1):
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)

    hs_tuple = output["hidden_states"]
    hs = hs_tuple[layer][0, -1].detach().cpu().numpy()

    return hs

# Extract LLaMA hidden states from a string of text.
# Optionally add an EOS token to the end of the input.
# Returns activations from the layer numbered `layer` (or -1 for last layer)
def llama_hs_from_text(model, tokenizer, text, layer=-1, add_eos=True):
    if add_eos:
      text = text + tokenizer.eos_token

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    return llama_hs_from_tokens(model, input_ids, layer)

s = llama_hs_from_text(model, tokenizer, "This string has a hidden state!")

import matplotlib.pyplot as plt
import pandas as pd

plt.subplot(311)
plt.title("a histogram")
plt.hist(hs,50,stacked=True, density=True)

import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from datasets import load_dataset

# Load the amazon polarity dataset.
# This is stored on google drive (!) and sometimes the download throws
# weird errors. You might need a new version of `datasets` or the drive
# folder might be at its bandwidth limit for the day lol.
data = load_dataset("amazon_polarity")["test"]

def format_amazon(text, label):
    return "A customer wrote the following review:\n{}\nThe sentiment in this review is {}.".format(text,  ["negative", "positive"][label])

def make_training_data(model, tokenizer, data, format=format_amazon, n=200):

    model.eval()
    all_neg_hs, all_pos_hs, all_gt_labels, all_text = [], [], [], []

    # loop
    for _ in tqdm(range(n)):
        # for simplicity, sample a random example until we find one that's a reasonable length
        # (most examples should be a reasonable length, so this is just to make sure)
        while True:
            idx = np.random.randint(len(data))
            text, true_label = data[idx]["content"], data[idx]["label"]
            # the actual formatted input will be longer, so include a bit of a marign
            if len(tokenizer(text)) < 400:
                break

        # get hidden states
        neg_hs = llama_hs_from_text(model, tokenizer, format_amazon(text, 0))
        pos_hs = llama_hs_from_text(model, tokenizer, format_amazon(text, 1))

        # collect
        all_neg_hs.append(neg_hs)
        all_pos_hs.append(pos_hs)
        all_gt_labels.append(true_label)
        all_text.append(text)

    all_neg_hs = np.stack(all_neg_hs)
    all_pos_hs = np.stack(all_pos_hs)
    all_gt_labels = np.stack(all_gt_labels)

    return all_neg_hs, all_pos_hs, all_gt_labels, all_text

# Make all training data
neg_hs, pos_hs, y, all_text = make_training_data(model, tokenizer, data, format=format_amazon, n=200)

# 50/50 train/test split
n = len(y)
neg_hs_train, neg_hs_test = neg_hs[:n//2], neg_hs[n//2:]
pos_hs_train, pos_hs_test = pos_hs[:n//2], pos_hs[n//2:]
text_train, text_test = all_text[:n//2], all_text[n//2:]
y_train, y_test = y[:n//2], y[n//2:]

# Try simple logistic regression to see if that works.
# Learn a plane that separates differences in the true direction vs differences
# in the false direction.

x_train = neg_hs_train - pos_hs_train
x_test = neg_hs_test - pos_hs_test

lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(x_train, y_train)
print("Logistic regression accuracy: {}".format(lr.score(x_test, y_test)))

# Subtract out the mean (and maybe also normalize variance) of a data set
def normalize(x, var_normalize = False):
  normalized_x = x - x.mean(axis=0, keepdims=True)
  if var_normalize:
      normalized_x /= normalized_x.std(axis=0, keepdims=True)

  return normalized_x

# Collin's main loss function
def informative_loss(p0, p1):
  return (torch.min(p0, p1)**2).mean(0)

def consistent_loss(p0, p1):
  return ((p0 - (1-p1))**2).mean(0)

def ccs_loss(p0, p1):
  return informative_loss(p0,p1) + consistent_loss(p0,p1)

def get_tensor_data(x0, x1):
  x0 = torch.tensor(x0, dtype=torch.float, requires_grad=False, device=model.device)
  x1 = torch.tensor(x1, dtype=torch.float, requires_grad=False, device=model.device)
  return x0, x1

# Learn the plane from CCS. For some reason this doesn't always converge so we'll
# do `ntries` runs. 10 seems to be more than enough.
# Returns [best_probe, best_loss]
def ccs(x0, x1, nepochs=1000, ntries=10, lr=1e-3, verbose=False, weight_decay=0.01, var_normalize=False, loss_func=ccs_loss):
    # Collin subtracts out the means before training
    x0 = normalize(x0, var_normalize=var_normalize)
    x1 = normalize(x1, var_normalize=var_normalize)

    # Number of entries in the hidden states
    d = x0.shape[-1]

    # Probe that we'll learn
    probe = nn.Sequential(nn.Linear(d, 1),nn.Sigmoid())
    probe.to(model.device)
    best_probe = copy.deepcopy(probe)

    best_loss = np.inf



    for train_num in range(ntries):
        # Make a new probe for this run
        probe = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
        probe.to(model.device)

        # Order the data randomly in a tensor
        x0, x1 = get_tensor_data(x0, x1)
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]

        # Set up optimizer. Collin uses adamW so that's what we'll go with
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

        # Start training
        for epoch in range(nepochs):
          # probe
          p0, p1 = probe(x0), probe(x1)

          # get the corresponding loss
          loss = loss_func(p0, p1)

          # update the parameters
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        loss = loss.detach().cpu().item()

        if verbose:
          print("Round {}: loss is {}".format(train_num, loss))

        if loss < best_loss:
            best_probe = copy.deepcopy(probe)
            best_loss = loss

    return best_probe, best_loss


def predict_pairs(probe, x0, x1):
  x0 = torch.tensor(x0, dtype=torch.float, requires_grad=False, device=model.device)
  x1 = torch.tensor(x1, dtype=torch.float, requires_grad=False, device=model.device)

  with torch.no_grad():
      p0, p1 = probe(x0), probe(x1)

  avg_confidence = p0 - p1
  predictions = (avg_confidence.detach().cpu().numpy())[:,0]
  return predictions

# Get raw credence scores (before sigmoid) from a single hidden state.
# `hs` is a numpy array of hidden states to apply this to
def classify_single(probe, hs):
  # Extract the actual vectors
  classifier_direction = np.squeeze(np.transpose(probe[0].weight.detach().cpu().numpy()))
  confidences = np.apply_along_axis(lambda x : np.dot(x,classifier_direction), 1, hs)

  return confidences

def get_acc(probe, x0_test, x1_test, y_test):
  predictions = (predict_pairs(probe, x0_test, x1_test) < 0.5).astype(int)

  # If predictions get messed up (i.e. ever not 1 or 0) this method will show
  # really good accuracy. TODO evaluate vs y_test and y_test inverted to avoid
  # this.
  acc = (predictions == y_test).mean()

  acc = max(acc, 1 - acc)

  return acc

def is_reversed(probe, x0_test, x1_test, y_test):
  predictions = (predict_pairs(probe, x0_test, x1_test) < 0.5).astype(int)

  # If predictions get messed up (i.e. ever not 1 or 0) this method will show
  # really good accuracy. TODO evaluate vs y_test and y_test inverted to avoid
  # this.
  acc = (predictions == y_test).mean()

  return acc < 0.5

probe, loss = ccs(neg_hs_train, pos_hs_train, ntries=3, verbose=True)

print("Learned probe:\n")
print(np.squeeze(np.transpose(probe[0].weight.detach().cpu().numpy())))
ccs_acc = get_acc(probe, neg_hs_test, pos_hs_test, y_test)

print("CCS Accuracy: {}, loss: {}".format(ccs_acc, loss))

classifier_direction = np.squeeze(np.transpose(probe[0].weight.detach().cpu().numpy()))

neg_credences = np.apply_along_axis(lambda x : np.dot(x,classifier_direction), 1, neg_hs_test)
pos_credences = np.apply_along_axis(lambda x : np.dot(x,classifier_direction), 1, pos_hs_test)


import matplotlib.pyplot as plt
import pandas as pd

plt.subplot(311)
plt.title("Raw scores (before normalize or sigmoid) from pos H.S.")
plt.hist([pos_credences[[(not not x) for x in y_test]], pos_credences[[not x for x in y_test]]], stacked=True, density=True)

plt.subplot(312)
plt.title("Raw scores (before normalize or sigmoid) from neg H.S.")
plt.hist([neg_credences[[(not not x) for x in y_test]], neg_credences[[not x for x in y_test]]], stacked=True, density=True)

plt.show()

# For playing around with random statements


# print(is_reversed(probe, neg_hs_test, pos_hs_test, y_test))

# def sig(x):
#  return 1/(1 + np.exp(-x))

# while(True):
#     t = input()
#     hs = get_llama_hidden_states(model, tokenizer, t)
#     raw = np.dot(classifier_direction, hs)
#     print("'{}' scores {} (sigmoid {})".format(t, raw, sig(raw)))

# Component of x perpendicular to y
def perp(x,y):
  along = y * (np.dot(x,y) / np.dot(y,y))
  return x - along

residual_neg_hs_train = neg_hs_train
residual_pos_hs_train = pos_hs_train

residual_neg_hs_test = neg_hs_test
residual_pos_hs_test = pos_hs_test

accs = []
train_accs = []
losses = []
probes = []


for i in range(5):
  probe, loss = ccs(residual_neg_hs_train, residual_pos_hs_train, ntries=5, loss_func=ccs_loss)

  ccs_acc = get_acc(probe, residual_neg_hs_test, residual_pos_hs_test, y_test)
  ccs_train_acc = get_acc(probe, residual_neg_hs_train, residual_pos_hs_train, y_train)

  print("CCS accuracy (component {}): {} in training, {} in testing, {} loss".format(i,ccs_train_acc,ccs_acc, loss))

  train_accs.append(ccs_train_acc)
  accs.append(ccs_acc)
  losses.append(loss)
  probes.append(probe)

  # The direction we just found that best classifies the data
  classifier_direction = np.squeeze(np.transpose(probe[0].weight.detach().cpu().numpy()))

  residual_neg_hs_train = np.apply_along_axis(lambda x : perp(x,classifier_direction), 1, residual_neg_hs_train)
  residual_pos_hs_train = np.apply_along_axis(lambda x : perp(x,classifier_direction), 1, residual_pos_hs_train)
  residual_neg_hs_test = np.apply_along_axis(lambda x : perp(x,classifier_direction), 1, residual_neg_hs_test)
  residual_pos_hs_ttest = np.apply_along_axis(lambda x : perp(x,classifier_direction), 1, residual_pos_hs_test)


import matplotlib.pyplot as plt
import pandas as pd

plt.plot(accs, label='Acc')
plt.plot(train_accs, label='Train Acc')
plt.plot(np.array(losses) * 10, label='Train CCS Loss * 10')
plt.legend()
plt.ylim(0,1)


plt.show()


preds1 = predict_pairs(probes[0], pos_hs_test, neg_hs_test)
preds2 =  predict_pairs(probes[1], pos_hs_test, neg_hs_test)
# preds3 =   predict(probes[1], pos_hs_test, neg_hs_test)

# preds4 = predict(probes[0], pos_hs_train, neg_hs_train)
# preds5 = predict(probes[0], pos_hs_train, neg_hs_train)
# preds6 = predict(probes[0], pos_hs_train, neg_hs_train)
plot_component_classification(preds1, preds2, y_test, text_test)


prompt = "The capital of the United States is Washington"
tokenizer.pad_token_id = tokenizer.eos_token_id
inputs = tokenizer([prompt], return_tensors="pt")

# Example 1: Print the scores for each token generated with Greedy Search
outputs = model.generate(inputs["input_ids"].to(model.device),
    return_dict_in_generate=True,
    output_scores=True,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=10
)

for x in outputs.sequences:
  print(tokenizer.decode(x))

transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, normalize_logits=False
)

print("Prediction")

hsp = llama_hs_from_text(model, tokenizer, "The following statement is true: " + tokenizer.decode(outputs.sequences[0]))
hsn = llama_hs_from_text(model, tokenizer, "The following statement is false: " + tokenizer.decode(outputs.sequences[0]))
print(hsp)
print(predict_pairs(probes[0], [hsp], [hsn]))
# input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
# encoder-decoder models, like BART or T5.
input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
generated_tokens = outputs.sequences[:, input_length:]
for tok, score in zip(generated_tokens[0], transition_scores[0]):
    # | token | token string | logits | probability
    print(f"{tok:5d} | {tokenizer.decode(tok):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}")



#generate("A student was asked how they")

probe0Reversed = is_reversed(probe, pos_hs_test, neg_hs_test, y_test)

print(probe0Reversed)

def product(prob, truth):
    return prob * truth

def false(prob, truth):
    return(1-truth)

def prob(prob, truth):
    return prob

def truth(prob, truth):
    return truth


def model_directed(text, merge, verbose=True):
    k = 10
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
      outputs = model(input_ids)

    next_token_logits = outputs["logits"]

    predictions = F.softmax(outputs.logits, dim=-1).detach().cpu()

    thresh = 1e-3
    vocab_size = predictions.shape[-1]

    # Predictions has one sentence (index 0) and we look at the last token predicted (-1)
    last_token_predictions = predictions[0][-1]

    idxs = torch.argsort(last_token_predictions, descending=True)[0:k]

    tokens = tokenizer.convert_ids_to_tokens(idxs)
    probs = last_token_predictions[idxs].tolist()

    clean_input_ids = input_ids.tolist()[0]

    next_states = [tokenizer.decode(clean_input_ids + [x.tolist()])[1:]  for x in idxs]

    hs = [llama_hs_from_text(model, tokenizer, x) for x in next_states]



    truthiness = classify_single(probe, hs)
    if(probe0Reversed):
      truthiness = 1-truthiness

    def sigmoid(x):
      return 1/(1 + np.exp(-3 * x))

    truthiness = truthiness - np.mean(truthiness)

    truthiness = [sigmoid(x) for x in truthiness]

    scores = torch.tensor([merge(probs[i], truthiness[i]) for i in range(k)])

    resorted = torch.argsort(scores, descending=True)

    if(verbose):
      print("\n")
      for i in range(k):
        print('{}: {}(prob={:.5f}, truth={:.5f}) = {} '.format(tokens[resorted[i]],score.__name__,probs[resorted[i]],truthiness[resorted[i]], scores[resorted[i]]))

    return next_states[resorted[0]]

# instruction = "Analyze the sentiment in the following statement. Output 'Negative' or 'Positive'.\nI hate the whole world so much; I detest everything."

# prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# {instruction}
# ### Response:"""

question = "Have humans flown close to the sun?"


prompt = """Sam4621: This was interesting. Thanks!

Alex1083: Glad I could help. Any more questions?

Sam4621: Yes, I do have a question. {0}

Alex1083:""".format(question)


outputs = []

for score in [prob, product, truth]:#[prob, product, truthHeavy, truthHeavier, truth, false, falser, falsest]:
  print("")
  text = prompt
  for i in range(20):
    text = model_directed(text, score, verbose=True)
    print("{} : {}".format(score.__name__,text.replace("\n","\\n")), end="\r")
  outputs.append("{} : {}".format(score.__name__,text.replace("\n","\\n")))

for x in outputs:
  print(x)


