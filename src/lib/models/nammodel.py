from tensorflow.keras import Model, layers, optimizers, metrics, models, activations

def create_NAM_model(L : int, no_entities : int, no_relations : int) -> Model:
    input_head = layers.Input((1,), dtype="int32", name="HeadInput")
    input_relation = layers.Input((1,), dtype="int32", name="RelationInput")
    input_tail = layers.Input((1,), dtype="int32", name="TailInput")

    relation_embed = layers.Embedding(no_relations, 100, input_length=1, name="RelationEmbedding")
    entity_embed = layers.Embedding(no_entities, 100, input_length=1, embeddings_initializer='zeros', name="EntityEmbedding")

    head_embed = entity_embed(input_head)
    tail_embed = entity_embed(input_tail)
    rela_embed = relation_embed(input_relation)

    x = layers.Concatenate()([head_embed, rela_embed])
    size = 100
    for l in range(L):
        x = layers.Dense(size, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

    out = layers.Dot(-1)([x, tail_embed])
    out = layers.Flatten()(out)
    out = activations.sigmoid(out)

    model = Model(inputs=[input_head, input_relation, input_tail], outputs=out, name="NeuralAssociationModel")

    opt = optimizers.Nadam(lr=0.01)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    return model