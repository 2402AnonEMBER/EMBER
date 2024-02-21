import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextIteratorStreamer, NoRepeatNGramLogitsProcessor, RepetitionPenaltyLogitsProcessor
from threading import Thread
import json
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="conll2003_generated", help="dataset")
parser.add_argument("--model", type=str, default="gpt2-xl", help="model")
args = parser.parse_args()

css = """
    <style>
    .highlight {
        display: inline;
    }
    .highlight::after {
        background-color: var(data-color);
    }
    .tooltip {
    position: relative;
    display: inline-block;
}

.tooltip::after {
    content: attr(data-tooltip-text); /* Set content from data-tooltip-text attribute */
    display: none;
    position: absolute;
    background-color: #333;
    color: #fff;
    padding: 5px;
    border-radius: 5px;
    bottom: 100%; /* Position it above the element */
    left: 50%;
    transform: translateX(-50%);
}

.tooltip:hover::after {
    display: block; /* Show the tooltip on hover */
}

    </style>"""

def map_value_to_color(value, colormap_name='tab20c'):
    """
    Map a value between 0 and 1 to a CSS color using a Python colormap.

    Args:
        value (float): A value between 0 and 1.
        colormap_name (str): The name of the colormap to use (e.g., 'viridis').

    Returns:
        str: A CSS color string in the form 'rgb(r, g, b)'.
    """
    # Ensure the value is within the range [0, 1]
    value = np.clip(value, 0.0, 1.0)

    # Get the colormap
    colormap = plt.get_cmap(colormap_name)

    # Map the value to a color
    rgba_color = colormap(value)

    # Convert the RGBA color to CSS format
    css_color = to_hex(rgba_color)

    return css_color + "88"


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, cuda=False):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)  # Input layer to hidden layer
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)  # Hidden layer to output layer
        if cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.to(self.device)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        return x

def get_iob_labels(dataset):
    if dataset == "conll2003":
        return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    elif dataset == "tner/ontonotes5":
        return ['O', 'B-CARDINAL', 'B-DATE', 'I-DATE', 'B-PERSON', 'I-PERSON', 'B-NORP', 'B-GPE', 'I-GPE', 'B-LAW', 'I-LAW', 'B-ORG', 'I-ORG', 'B-PERCENT', 'I-PERCENT', 'B-ORDINAL', 'B-MONEY', 'I-MONEY', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-FAC', 'B-TIME', 'I-CARDINAL', 'B-LOC', 'B-QUANTITY', 'I-QUANTITY', 'I-NORP', 'I-LOC', 'B-PRODUCT', 'I-TIME', 'B-EVENT', 'I-EVENT', 'I-FAC', 'B-LANGUAGE', 'I-PRODUCT', 'I-ORDINAL', 'I-LANGUAGE']
    else:
        raise NotImplementedError


def iob_to_classwise(preds, dataset_id, first_only=False):
    map = []

    new_tags = sorted(set([x.split("-")[-1] for x in get_iob_labels(dataset_id)]), key=[x.split("-")[-1] for x in get_iob_labels(dataset_id)].index)

    tags = get_iob_labels(dataset_id)
    for i, tag in enumerate(tags):
        if tag.startswith("I-") or tag.startswith("B-"):
            map.append(new_tags.index(tag[2:]))
        else:
            map.append(i)

    preds_new = torch.zeros_like(torch.Tensor(preds))

    for i, j in enumerate(map):
        preds_new[preds == i] = j

    return preds_new

@st.cache_resource
def get_classifiers_for_model(model_name, device, dataset):
    classifier_token = None
    config = json.load(open(f"checkpoints/{model_name}/config.json", "r"))
    config = config[dataset]
    #print(config)

    layer_id = config["layer"]
    
    classifier_span = MLP(model.config.n_head*model.config.n_layer, 2, hidden_dim=config["dim_span"]).to(device)
    classifier_span.load_state_dict(torch.load(f"checkpoints/{model_name}/{config['span']}", map_location=device))

    classifier_token = MLP(model.config.n_embd, config["n_classes"], hidden_dim=config["dim_span"]).to(device)
    classifier_token.load_state_dict(torch.load(f"checkpoints/{model_name}/{config['tokenwise']}", map_location=device))

    print(sum(p.numel() for p in classifier_span.parameters()), sum(p.numel() for p in classifier_token.parameters()))

    return classifier_span, classifier_token, layer_id

@st.cache_resource
def get_model(model_name, cuda=False):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, use_fast=True, cache_dir="cache/")
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir="cache/")
    if cuda:
        model.to("cuda:0").half()
    return tokenizer, model


# Load pre-trained GPT-2 model and tokenizer
model_name = args.model
tokenizer, model = get_model(model_name, cuda=False)
dataset = args.dataset
classifier_span, classifier_token, layer_id = get_classifiers_for_model(model_name, model.device, dataset)

if dataset == "conll2003_generated":
    dataset = "conll2003"

# Streamlit app
def main():
    st.title("EMBER system demonstration")

    # Input text prompt
    prompt = st.text_area("Enter your prompt:", "")

    # Output text field
    output_field = st.empty()
    output_text = ""
    output_labels = ""
    tokens = []
    tokenwise_preds = []
    spans = []

    new_tags = sorted(set([x.split("-")[-1] for x in get_iob_labels(dataset)]), key=[x.split("-")[-1] for x in get_iob_labels(dataset)].index)

    if st.button("Generate"):
        if not prompt:
            st.warning("Please enter a prompt.")
        else:
            # Generate text using streaming
            streamer = greedy_decoding_generator(model, tokenizer, prompt)
            # Display the generated text
            for chunk in streamer:
                output_field.empty()
                with output_field.container():

                    
                    tokens.extend(chunk[2])
                    spans.extend(chunk[3])
                    tokenwise_preds.extend(chunk[1])

                    # tokenwise annotated text
                    annotated = []
                    for token, pred in zip(tokenizer.convert_ids_to_tokens(tokens), tokenwise_preds):
                        predicted_class = ""
                        style = ""
                        if pred != 0:
                            predicted_class = f"tooltip highlight"
                            style = f"background-color: {map_value_to_color((pred-1)/(len(new_tags)-1))}"
                            if tokenizer.convert_tokens_to_string([token]).startswith(" "):
                                annotated.append("Ġ")
                            annotated.extend([f"<span class='{predicted_class}' data-tooltip-text='{new_tags[pred]}' style='{style}'>".replace(" ", "Ġ"), token.replace(" ", "Ġ"), "</span>"])
                        else:
                            annotated.append(token)

                    generated_text_tokenwise = tokenizer.convert_tokens_to_string(annotated).replace("<|endoftext|>", "")

                    # spanwise annotated text
                    annotated = []
                    span_ends = -1
                    in_span = False

                    out_of_span_tokens = []
                    token_strings = tokenizer.convert_ids_to_tokens(tokens)
                    for i in reversed(range(len(tokenwise_preds))):

                        if in_span:
                            if i >= span_ends:
                                continue
                            else:
                                in_span = False

                        predicted_class = ""
                        style = ""

                        span = None
                        for s in spans:
                            if s[1] == i:
                                span = s

                        if tokenwise_preds[i] != 0 and span is not None:
                            predicted_class = f"tooltip highlight"
                            style = f"background-color: {map_value_to_color((tokenwise_preds[i]-1)/(len(new_tags)-1))}"
                            if tokenizer.convert_tokens_to_string([token]).startswith(" "):
                                annotated.append("Ġ")
                            span_opener = f"Ġ<span class='{predicted_class}' data-tooltip-text='{new_tags[tokenwise_preds[i]]}' style='{style}'>".replace(" ", "Ġ")
                            span_end = "</span>"
                            annotated.extend(out_of_span_tokens)
                            out_of_span_tokens = []
                            span_ends = span[0]
                            in_span = True
                            annotated.append(span_end)
                            annotated.extend([token_strings[x] for x in reversed(range(span[0], span[1]+1))])
                            annotated.append(span_opener)
                        else:
                            out_of_span_tokens.append(token_strings[i])

                    annotated.extend(out_of_span_tokens)
                    generated_text_spanwise = tokenizer.convert_tokens_to_string(reversed(annotated)).replace("<|endoftext|>", "")
                    output_text = css + "<h3>Tokenwise classification</h3>" + generated_text_tokenwise + "\n\n" + "<h3>Spanwise label propagation</h3>" + generated_text_spanwise + "<br>"
                    #output_text = css + generated_text_spanwise

                    st.markdown(output_text, unsafe_allow_html=True)

def filter_spans(spans, values):
    # Create a dictionary to store spans based on their second index values
    span_dict = {}

    # Iterate through the spans and update the dictionary with the highest value
    for span, value in zip(spans, values):
        start, end = span
        current_value = span_dict.get(end, None)

        if current_value is None or current_value[1] < value:
            span_dict[end] = (span, value)

    # Extract the filtered spans and values
    filtered_spans, filtered_values = zip(*span_dict.values())

    return list(filtered_spans), list(filtered_values)

def greedy_decoding_generator(model, tokenizer, input_text, max_steps=100):
    input_text = "<|endoftext|> " + input_text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    past_key_values = None

    prev_text = input_text
    offset = 0
    all_ids = []

    for _ in range(max_steps):
        with torch.no_grad():

            outputs = model(input_ids.to(model.device), past_key_values=past_key_values, output_hidden_states=True, output_attentions=True)

            
            logitsprocessor = RepetitionPenaltyLogitsProcessor(1.2)
            all_ids_tensor = torch.tensor([all_ids])
            modified_logits = logitsprocessor(all_ids_tensor, outputs.logits[:, -1, :])
            # print(all_ids_tensor.shape, outputs.logits[:, -1, :].shape)
            next_token = torch.argmax(modified_logits).unsqueeze(0).unsqueeze(0)

            
            # decoded_output = tokenizer.decode(next_token[0], skip_special_tokens=True)
            decoded_output = ""
            past_key_values = outputs.past_key_values


            # tokenwise prediction
            h_s_at_layer = outputs.hidden_states[layer_id][0]
            pred_ner = classifier_token(h_s_at_layer.to(classifier_token.fc1.weight.dtype))
            pred_ner = iob_to_classwise(torch.argmax(pred_ner, dim=-1), dataset).cpu().tolist()

            attentions = torch.stack(outputs.attentions).swapaxes(0,1)[0]
            shape = attentions.shape
            attentions = attentions.view(-1, attentions.size(-2), attentions.size(-1))
            attentions = attentions.view(-1, attentions.size(-2)*attentions.size(-1))
            attentions = attentions.swapaxes(0,1)

            pred_softmax = torch.softmax(classifier_span(attentions.to(classifier_span.fc1.weight.dtype)), dim=-1)
            pred_span = torch.argmax(pred_softmax, dim=-1)

            pred_softmax = pred_softmax[:,-1].view(shape[-2], shape[-1])
            pred_span = pred_span.view(shape[-2], shape[-1])
            spans = (pred_span == 1).nonzero(as_tuple=True)
            values = []
            if len(spans) == 2:
                values = [pred_softmax[x,y] for x,y in zip(*spans)]
                spans = [(int(x), offset + int(y) - 1) for y,x in zip(*spans)]
                if len(spans) > 0:
                    # filter out spans that share the same end index
                    spans, values = filter_spans(spans, values)
            else:
                spans = []
            offset += input_ids.size(-1)

            yield (prev_text, pred_ner, input_ids[0], spans)

            all_ids.extend(input_ids[0])
            prev_text = decoded_output
            input_ids = next_token


if __name__ == "__main__":
    main()
