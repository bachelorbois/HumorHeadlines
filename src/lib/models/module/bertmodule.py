from bert import params_from_pretrained_ckpt, BertModelLayer

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
    return BertModelLayer.from_params(params)