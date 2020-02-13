from bert import params_from_pretrained_ckpt, BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from tensorflow.keras import layers

def get_bert_config(model_dir):
    """Function to get the bert config params
    
    Arguments:
        model_dir {String} -- Path to the bert_config.json file
    """
    return params_from_pretrained_ckpt(model_dir)


def get_bert_layer(params, name="BERT"):
    """Get the BERT layer from a set of specific parameters
    
    Arguments:
        params {BERT Params} -- Parameters for the BERT model. Grab them using get_bert_config
    
    Keyword Arguments:
        name {str} -- Name of the model (default: {"BERT"})
    
    Returns:
        BertModelLayer -- Layer to place in our model
    """
    return BertModelLayer.from_params(params, name=name)

def get_adapter_BERT_layer(model_dir, adapter_size):
    """Create a adapter-BERT layer
    
    Arguments:
        model_dir {str} -- Path to the pretrained model files
        adapter_size {int} -- Size of adapter
    
    Returns:
        BERT -- BERT layer
    """
    with open(model_dir+'/bert_config.json', 'r') as fd:
        bc = StockBertConfig.from_json_string(fd.read())
        params = map_stock_config_to_params(bc)
        params.adapter_size = adapter_size
    return get_bert_layer(params)

def flatten_layers(root_layer):
    if isinstance(root_layer, layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer


def freeze_bert_layers(l_bert):
    """Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    
    Arguments:
        l_bert {Layer} -- adapter-BERT layer
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False