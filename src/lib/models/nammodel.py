from tensorflow.keras import Model, layers, optimizers, metrics, models

def create_NAM_model() -> Model:
    input_head = layers.Input((1,), dtype="int32", name="HeadInput")
    input_relation = layers.Input((1,), dtype="int32", name="RelationInput")
    input_tail = layers.Input((1,), dtype="int32", name="TailInput")

    relation_embed = layers.Embedding()
    entity_embed = layers.Embedding()
