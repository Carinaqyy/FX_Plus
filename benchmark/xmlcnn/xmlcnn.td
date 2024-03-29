Passes<list> {
    LossEliminate<pass> {
        src: fx_plus.compiler.passes;
    }
    FrontendPass<pass> {
        src: fx_plus.compiler.passes;
    },
    DecompositionPass<pass> {
        src: fx_plus.compiler.passes;
    },
    LocalCSE<pass> {
        src: fx_plus.compiler.passes
    },
    Reassociation<pass> {
        src: fx_plus.compiler.passes;
    },
    TritonFusionPass<pass> {
        src: fx_plus.compiler.passes;
    }
}

XMLCNN<Model> {
    init_args<list> {
        config<attr_cls> {
            sequence_length<int> {
                description: sequence length of the inputs;
            }

            embedding_dim<int>{ description: the dimension of the embedding; }
            filter_sizes<list>{
                description: a list containing the size of filters;
            }
            num_filters<int>  { 
                description: number of output channels of convolution layers;
            }
            hidden_dims<int>  {description: fc layer's hidden dimension;}
            y_dim<int> {description: number of classification classes; }
        }
    }

    runtime_args<list> {
        batch_size<int> {description: batch size of the input; }
    }

    inputs <list> {
        e_emb<tensor> {
            distrib: randn;
            size: (batch_size, sequence_length, embedding_dim);
            dtype: float16;
            description: input embedding;
        }
        
        y<tensor> {
            distrib: randint;
            low: 0;
            high: 2;
            size: (batch_size, y_dim);
            dtype: float16;
            description: binary labels;
        }
    }

    dtype: float16;

    description: the XML-CNN comes from the paper Deep Learning for Extreme 
        Multi-label Text Classification with dynamic pooling;
}
